/* Copyright (c) 2018-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "Vasculature.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

#include <plugin/io/db/DBConnector.h>

#include <brayns/common/Timer.h>
#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/parameters/ParametersManager.h>

namespace bioexplorer
{
namespace vasculature
{
using namespace common;
using namespace io;
using namespace db;

Vasculature::Vasculature(Scene& scene, const VasculatureDetails& details)
    : _details(details)
    , _scene(scene)
{
    Timer chrono;
    _importFromDB();
    _buildModel();
    PLUGIN_TIMER(chrono.elapsed(), "Vasculature loaded");
}

void Vasculature::_importFromDB()
{
    auto& connector = DBConnector::getInstance();

    _nodes = connector.getVasculatureNodes(_details.populationName,
                                           _details.sqlFilter);

    PLUGIN_INFO(1, "Full vasculature is made of " << _nodes.size() << " nodes");

    std::map<uint64_t, std::vector<uint64_t>> pairs;
    std::set<uint64_t> entryNodes;
    const auto& gids = _details.gids;
    for (const auto& node : _nodes)
    {
        const auto sectionId = node.second.sectionId;
        if (!gids.empty() &&
            // Load specified edges only
            std::find(gids.begin(), gids.end(), sectionId) == gids.end())
            continue;

        _graphs.insert(node.second.graphId);
        pairs[node.second.pairId].push_back(node.first);
        _sectionIds.insert(node.second.sectionId);
        _sections[sectionId].push_back(node.first);
        entryNodes.insert(node.second.entryNodeId);
    }

    for (const auto& pair : pairs)
    {
        if (pair.second.size() != 2)
            PLUGIN_WARN("Invalid number of Ids for pair " << pair.first);

        const auto nbIds = pair.second.size();
        auto& edge1 = _nodes[pair.second[0]];
        edge1.pairId = pair.second[nbIds - 1];
        auto& edge2 = _nodes[pair.second[nbIds - 1]];
        edge2.pairId = pair.second[nbIds - 1];
        const auto graphId1 = edge1.graphId;
        const auto graphId2 = edge2.graphId;
        for (auto& node : _nodes)
            if (node.second.graphId == graphId1 ||
                node.second.graphId == graphId2)
                node.second.pairId = pair.first;
    }

    for (const auto& section : _sections)
        _nbMaxPointsPerSection =
            std::max(_nbMaxPointsPerSection, section.second.size());

    _nbPairs = pairs.size();
    _nbEntryNodes = entryNodes.size();
    PLUGIN_INFO(1,
                "Loaded vasculature is made of " << _nodes.size() << " nodes");
}

std::set<uint64_t> Vasculature::_buildGraphModel(
    Model& model, const VasculatureColorSchemeDetails& details)
{
    std::set<uint64_t> materialIds;
    const float radiusMultiplier = _details.radiusMultiplier;
    size_t materialId = 0;
    for (const auto& section : _sections)
    {
        const auto& srcNode = _nodes[section.second[0]];
        const auto& dstNode = _nodes[section.second[section.second.size() - 1]];

        switch (details.colorScheme)
        {
        case VasculatureColorScheme::section:
            materialId = dstNode.sectionId;
            break;
        case VasculatureColorScheme::subgraph:
            materialId = dstNode.graphId;
            break;
        case VasculatureColorScheme::pair:
            materialId = dstNode.pairId;
            break;
        case VasculatureColorScheme::entry_node:
            materialId = dstNode.entryNodeId;
            break;
        }

        const auto& src = srcNode.position;
        const auto& dst = dstNode.position;
        const auto userData = section.second[0];
        const auto direction = dst - src;
        const auto maxRadius = std::max(srcNode.radius, dstNode.radius);
        const float radius =
            std::min(length(direction) / 5.0, maxRadius * radiusMultiplier);
        model.addSphere(materialId, {src, radius * 0.2f, userData});
        model.addCylinder(materialId, {src, Vector3f(src + direction * 0.79),
                                       radius * 0.2f, userData});
        model.addCone(materialId, {dst, Vector3f(src + direction * 0.8), 0.0,
                                   radius, userData});
        model.addCone(materialId, {Vector3f(src + direction * 0.8),
                                   Vector3f(src + direction * 0.79), radius,
                                   radius * 0.2f, userData});
        materialIds.insert(materialId);
    }
    return materialIds;
}

std::set<uint64_t> Vasculature::_buildSimpleModel(
    Model& model, const VasculatureColorSchemeDetails& details)
{
    std::set<uint64_t> materialIds;
    SDFMorphologyData sdfMorphologyData;
    const float radiusMultiplier = _details.radiusMultiplier;
    const auto useSdf = _details.useSdf;
    size_t materialId = 0;
    size_t previousMaterialId = 0;
    for (const auto& section : _sections)
    {
        for (uint64_t i = 0; i < section.second.size(); ++i)
        {
            const auto& srcNode = _nodes[section.second[i]];
            switch (details.colorScheme)
            {
            case VasculatureColorScheme::section:
                materialId = srcNode.sectionId;
                break;
            case VasculatureColorScheme::subgraph:
                materialId = srcNode.graphId;
                break;
            case VasculatureColorScheme::node:
                materialId = section.second[i];
                break;
            case VasculatureColorScheme::pair:
                materialId = srcNode.pairId;
                break;
            case VasculatureColorScheme::entry_node:
                materialId = srcNode.entryNodeId;
                break;
            case VasculatureColorScheme::section_gradient:
                materialId = i * double(_nbMaxPointsPerSection) /
                             double(section.second.size());
                break;
            }

            const auto& src = srcNode.position;
            const auto srcRadius = srcNode.radius * radiusMultiplier;
            const auto userData = section.second[i];
            const auto sectionId = srcNode.sectionId;
            _addStepSphereGeometry(useSdf, src, srcRadius, materialId, userData,
                                   model, sdfMorphologyData, sectionId);
            if (i > 0)
            {
                const auto& dstNode = _nodes[section.second[i - 1]];
                const auto& dst = dstNode.position;
                const auto dstRadius = dstNode.radius * radiusMultiplier;
                _addStepConeGeometry(useSdf, dst, dstRadius, src, srcRadius,
                                     previousMaterialId, userData, model,
                                     sdfMorphologyData, sectionId);
            }
            previousMaterialId = materialId;
            materialIds.insert(materialId);
        }
    }
    if (_details.useSdf)
        _finalizeSDFGeometries(model, sdfMorphologyData);

    return materialIds;
}

std::set<uint64_t> Vasculature::_buildAdvancedModel(
    Model& model, const VasculatureColorSchemeDetails& details)

{
    std::set<uint64_t> materialIds;
    SDFMorphologyData sdfMorphologyData;
    const float radiusMultiplier = _details.radiusMultiplier;
    const auto useSdf = _details.useSdf;
    size_t materialId = 0;
    for (const auto& section : _sections)
    {
        Vector4ds controlPoints;
        for (const auto& nodeId : section.second)
        {
            const auto& node = _nodes[nodeId];
            controlPoints.push_back({node.position.x, node.position.y,
                                     node.position.z,
                                     node.radius * radiusMultiplier});
        }
        const size_t precision = 1;
        const double step = precision * 1.0 / double(section.second.size());
        uint64_t i = 0;
        for (double t = 0.0; t < 1.0 - step * 2.0; t += step)
        {
            const auto& srcNode = _nodes[section.second[i]];
            switch (details.colorScheme)
            {
            case VasculatureColorScheme::section:
                materialId = srcNode.sectionId;
                break;
            case VasculatureColorScheme::subgraph:
                materialId = srcNode.graphId;
                break;
            case VasculatureColorScheme::node:
                materialId = section.second[i];
                break;
            case VasculatureColorScheme::pair:
                materialId = srcNode.pairId;
                break;
            case VasculatureColorScheme::entry_node:
                materialId = srcNode.entryNodeId;
                break;
            case VasculatureColorScheme::section_gradient:
                materialId = i * double(_nbMaxPointsPerSection) /
                             double(section.second.size());
                break;
            }

            const auto userData = section.second[i];
            const Vector4f src = getBezierPoint(controlPoints, t);
            const auto sectionId = srcNode.sectionId;
            _addStepSphereGeometry(useSdf, Vector3f(src), src.w, materialId,
                                   userData, model, sdfMorphologyData,
                                   sectionId);
            if (i > 0)
            {
                const auto& dstNode = _nodes[section.second[i + 1]];
                const Vector4f dst = getBezierPoint(controlPoints, t + step);
                _addStepConeGeometry(useSdf, Vector3f(dst), dst.w,
                                     Vector3f(src), src.w, materialId, userData,
                                     model, sdfMorphologyData, sectionId);
            }
            materialIds.insert(materialId);
            i += precision;
        }
    }
    if (_details.useSdf)
        _finalizeSDFGeometries(model, sdfMorphologyData);

    return materialIds;
}

void Vasculature::_buildModel(const VasculatureColorSchemeDetails& details)
{
    auto model = _scene.createModel();
    std::set<uint64_t> materialIds;

    switch (_details.quality)
    {
    case VasculatureQuality::low:
        materialIds = _buildGraphModel(*model, details);
        break;
    case VasculatureQuality::medium:
        materialIds = _buildSimpleModel(*model, details);
        break;
    default:
        materialIds = _buildAdvancedModel(*model, details);
        break;
    }

#if 0
    const auto& connector = DBConnector::getInstance();
    const auto bifurcations =
        connector.getVasculatureBifurcations(_details.populationName);

    const size_t materialId = 0;
    for (const auto& bifurcation : bifurcations)
    {
        const auto& srcNode = _nodes[bifurcation.first];
        const Vector3f src = srcNode.position;
        const float srcRadius = srcNode.radius;
        for (const auto& nodeId : bifurcation.second)
        {
            const auto& dstNode = _nodes[nodeId];
            const Vector3f dst = dstNode.position;
            const float dstRadius = dstNode.radius;
            const auto userData = nodeId;
            model->addSphere(materialId, {dst, dstRadius, userData});
            model->addCone(materialId,
                           {src, dst, srcRadius, dstRadius, userData});
        }
    }
#endif

    uint64_t colorCount = 0;
    auto& palette = details.palette;
    for (const auto materialId : materialIds)
    {
        Vector3f color{1.f, 1.f, 1.f};
        if (!palette.empty())
        {
            color = Vector3f(palette[colorCount], palette[colorCount + 1],
                             palette[colorCount + 2]);
            colorCount += 3;
        }
        auto nodeMaterial =
            model->createMaterial(materialId, std::to_string(materialId));
        nodeMaterial->setDiffuseColor(color);
        nodeMaterial->setSpecularColor(color);
        nodeMaterial->setSpecularExponent(100.f);
        PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE, 0});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        props.setProperty({MATERIAL_PROPERTY_CHAMELEON_MODE, 0});
        props.setProperty({MATERIAL_PROPERTY_CAST_USER_DATA, false});
        props.setProperty({MATERIAL_PROPERTY_NODE_ID, 0});
        nodeMaterial->updateProperties(props);
    }

    ModelMetadata metadata = {
        {"Number of edges", std::to_string(_nodes.size())},
        {"Number of sections", std::to_string(_sectionIds.size())},
        {"Number of sub-graphs", std::to_string(_graphs.size())}};

    _modelDescriptor.reset(new brayns::ModelDescriptor(std::move(model),
                                                       _details.assemblyName,
                                                       metadata));
    if (_modelDescriptor)
        _scene.addModel(_modelDescriptor);
    else
        PLUGIN_THROW("Vasculature model could not be created for population " +
                     _details.populationName);
}

void Vasculature::setColorScheme(const VasculatureColorSchemeDetails& details)
{
    _buildModel(details);
}

void Vasculature::setRadiusReport(const VasculatureRadiusReportDetails& details)
{
    auto& connector = DBConnector::getInstance();
    const auto simulationReport =
        connector.getVasculatureSimulationReport(details.populationName,
                                                 details.simulationReportId);

    const size_t nbFrames =
        (simulationReport.endTime - simulationReport.startTime) /
        simulationReport.timeStep;
    if (nbFrames == 0)
        PLUGIN_THROW("Report does not contain any simulation data: " +
                     simulationReport.description);

    if (details.frame >= nbFrames)
        PLUGIN_THROW("Invalid frame specified for report: " +
                     simulationReport.description);
    const floats series =
        connector.getVasculatureSimulationTimeSeries(details.simulationReportId,
                                                     details.frame);

    auto& model = _modelDescriptor->getModel();
    auto& spheresMap = model.getSpheres();
    for (auto& spheres : spheresMap)
        for (auto& sphere : spheres.second)
            sphere.radius = details.amplitude * series[sphere.userData];

    auto& conesMap = model.getCones();
    for (auto& cones : conesMap)
        for (auto& cone : cones.second)
        {
            cone.centerRadius = details.amplitude * series[cone.userData];
            cone.upRadius = details.amplitude * series[cone.userData + 1];
        }

    auto& cylindersMap = model.getCylinders();
    for (auto& cylinders : cylindersMap)
        for (auto& cylinder : cylinders.second)
            cylinder.radius = details.amplitude * series[cylinder.userData];

    model.commitGeometry();
    model.updateBounds();
    PLUGIN_DEBUG("Vasculature geometry successfully modified using report "
                 << simulationReport.description);
    _scene.markModified(false);
}

} // namespace vasculature
} // namespace bioexplorer
