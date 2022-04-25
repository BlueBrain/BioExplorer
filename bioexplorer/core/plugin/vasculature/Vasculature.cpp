/* Copyright (c) 2020-2022, EPFL/Blue Brain Project
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
    : SDFGeometries(details.radiusMultiplier == 0 ? 1.0
                                                  : details.radiusMultiplier,
                    doublesToVector3d(details.scale))
    , _details(details)
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

    std::map<uint64_t, uint64_ts> pairs;
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
        _sectionIds.insert(node.second.sectionId);
        _sections[sectionId].push_back(node.first);

        pairs[node.second.pairId].push_back(node.first);
        entryNodes.insert(node.second.entryNodeId);
    }

    for (const auto& pair : pairs)
    {
        if (pair.second.size() != 2)
            PLUGIN_WARN("Invalid number of Ids for pair "
                        << pair.first << ". Expected 2, found "
                        << pair.second.size());

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
        _nbMaxSegmentsPerSection =
            std::max(_nbMaxSegmentsPerSection, section.second.size());

    _nbPairs = pairs.size();
    _nbEntryNodes = entryNodes.size();
}

void Vasculature::_applyPaletteToModel(Model& model, const doubles& palette)
{
    if (!palette.empty())
    {
        auto& materials = model.getMaterials();
        if (palette.size() / 3 < materials.size())
            // Note that the size of the palette can be greater than the number
            // of materials, depending on how optimized the model is. When being
            // constructed, some parts of the model might be removed because
            // they are so small that they become invisible
            PLUGIN_THROW("Invalid palette size. Expected " +
                         std::to_string(materials.size()) +
                         ", provided: " + std::to_string(palette.size() / 3));
        uint64_t i = 0;
        for (auto material : materials)
        {
            const Vector3f color{palette[i], palette[i + 1], palette[i + 2]};
            material.second->setDiffuseColor(color);
            material.second->setSpecularColor(color);
            i += 3;
        }
    }
}

void Vasculature::_buildGraphModel(Model& model,
                                   const VasculatureColorSchemeDetails& details)
{
    const double radiusMultiplier = _details.radiusMultiplier;
    size_t materialId = 0;
    ThreadSafeContainers containers;
    uint64_t index;
#pragma omp parallel for private(index)
    for (index = 0; index < _sections.size(); ++index)
    {
        if (omp_get_thread_num() == 0)
            PLUGIN_PROGRESS("Loading vasculature", index, _sections.size());

        auto it = _sections.begin();
        std::advance(it, index);
        const auto& section = it->second;
        const auto sectionId = it->first;

        ThreadSafeContainer container(model, _details.useSdf,
                                      doublesToVector3d(_details.scale));

        const auto& srcNode = _nodes[section[0]];
        const auto& dstNode = _nodes[section[section.size() - 1]];

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
        const auto userData = section[0];
        const auto direction = dst - src;
        const auto maxRadius = std::max(srcNode.radius, dstNode.radius);
        const float radius =
            std::min(length(direction) / 5.0, maxRadius * radiusMultiplier);
        container.addSphere(src, radius * 0.2, materialId, userData);
        container.addCone(src, radius * 0.2, Vector3f(src + direction * 0.79),
                          radius * 0.2, materialId, userData);
        container.addCone(dst, 0.0, Vector3f(src + direction * 0.8), radius,
                          materialId, userData);
        container.addCone(Vector3f(src + direction * 0.8), radius,
                          Vector3f(src + direction * 0.79), radius * 0.2,
                          materialId, userData);
#pragma omp critical
        containers.push_back(container);
    }

    for (size_t i = 0; i < containers.size(); ++i)
    {
        const float progress = 1.f + i;
        PLUGIN_PROGRESS("- Compiling 3D geometry...", progress,
                        containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }
}

void Vasculature::_buildSimpleModel(
    Model& model, const VasculatureColorSchemeDetails& details,
    const doubles& radii)
{
    const auto radiusMultiplier = _details.radiusMultiplier;
    const auto useSdf = _details.useSdf;

    ThreadSafeContainers containers;
    uint64_t index;
#pragma omp parallel for private(index)
    for (index = 0; index < _sections.size(); ++index)
    {
        if (omp_get_thread_num() == 0)
            PLUGIN_PROGRESS("Loading vasculature", index,
                            _sections.size() / omp_get_max_threads());

        auto it = _sections.begin();
        std::advance(it, index);
        const auto& section = it->second;
        const auto sectionId = it->first;

        ThreadSafeContainer container(model, _details.useSdf,
                                      doublesToVector3d(_details.scale));

        uint64_t geometryIndex = 0;
        size_t materialId = 0;
        size_t previousMaterialId = 0;
        Neighbours neighbours{};
        for (uint64_t i = 0; i < section.size() - 1; ++i)
        {
            const auto srcNodeId = section[i];
            const auto& srcNode = _nodes[srcNodeId];
            const auto userData = srcNodeId;

            switch (details.colorScheme)
            {
            case VasculatureColorScheme::section:
                materialId = srcNode.sectionId;
                break;
            case VasculatureColorScheme::subgraph:
                materialId = srcNode.graphId;
                break;
            case VasculatureColorScheme::node:
                materialId = srcNodeId;
                break;
            case VasculatureColorScheme::pair:
                materialId = srcNode.pairId;
                break;
            case VasculatureColorScheme::entry_node:
                materialId = srcNode.entryNodeId;
                break;
            case VasculatureColorScheme::section_gradient:
                materialId = i * double(_nbMaxSegmentsPerSection) /
                             double(section.size());
                break;
            }

            const auto& srcPoint = srcNode.position;
            const auto srcRadius =
                (userData < radii.size() ? radii[userData] : srcNode.radius) *
                radiusMultiplier;
            const auto sectionId = srcNode.sectionId;

            if (i == 0)
            {
                if (!useSdf)
                    container.addSphere(srcPoint, srcRadius, materialId,
                                        userData);
                previousMaterialId = materialId;
            }

            // Ignore points that are too close the previous one
            // (according to respective radii)
            Vector3d dstPoint;
            double dstRadius;
            do
            {
                const auto dstNodeId = section[i + 1];
                const auto& dstNode = _nodes[dstNodeId];
                dstPoint = dstNode.position;
                const auto dstUserData = dstNodeId;
                dstRadius = (dstUserData < radii.size() ? radii[dstUserData]
                                                        : dstNode.radius) *
                            radiusMultiplier;
                ++i;

            } while (length(dstPoint - srcPoint) < (srcRadius + dstRadius) &&
                     i < section.size() - 1);
            --i;

            if (!useSdf)
                container.addSphere(dstPoint, dstRadius, materialId, userData);

            geometryIndex =
                container.addCone(dstPoint, dstRadius, srcPoint, srcRadius,
                                  previousMaterialId, userData, neighbours,
                                  Vector3f(segmentDisplacementStrength,
                                           segmentDisplacementFrequency, 0.f));
            previousMaterialId = materialId;

            neighbours = {geometryIndex};
        }
#pragma omp critical
        containers.push_back(container);
    }
    PLUGIN_INFO(1, "");

    for (size_t i = 0; i < containers.size(); ++i)
    {
        PLUGIN_PROGRESS("- Compiling 3D geometry...", 1 + i, containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }
    PLUGIN_INFO(1, "");

    _applyPaletteToModel(model, details.palette);

    PLUGIN_ERROR("Created " + std::to_string(model.getMaterials().size()) +
                 " materials");
}

void Vasculature::_buildAdvancedModel(
    Model& model, const VasculatureColorSchemeDetails& details,
    const doubles& radii)
{
    const auto radiusMultiplier = _details.radiusMultiplier;
    const auto useSdf = _details.useSdf;
    size_t materialId = 0;

    std::vector<ThreadSafeContainer> containers;
    uint64_t index;
#pragma omp parallel for private(index)
    for (index = 0; index < _sections.size(); ++index)
    {
        if (omp_get_thread_num() == 0)
            PLUGIN_PROGRESS("Loading vasculature", index, _sections.size());

        auto it = _sections.begin();
        std::advance(it, index);
        const auto& section = it->second;
        const auto sectionId = it->first;

        ThreadSafeContainer container(model, _details.useSdf,
                                      doublesToVector3d(_details.scale));

        Vector4fs controlPoints;
        for (const auto& nodeId : section)
        {
            const auto& node = _nodes[nodeId];
            controlPoints.push_back({node.position.x, node.position.y,
                                     node.position.z,
                                     node.radius * radiusMultiplier});
        }

        const double step = 2.0 / double(section.size());
        uint64_t geometryIndex = 0;
        Neighbours neighbours;
        uint64_t i = 0;
        for (double t = 0.0; t < 1.0 - step * 2.0; t += step)
        {
            const auto& srcNode = _nodes[section[i]];
            switch (details.colorScheme)
            {
            case VasculatureColorScheme::section:
                materialId = srcNode.sectionId;
                break;
            case VasculatureColorScheme::subgraph:
                materialId = srcNode.graphId;
                break;
            case VasculatureColorScheme::node:
                materialId = section[i];
                break;
            case VasculatureColorScheme::pair:
                materialId = srcNode.pairId;
                break;
            case VasculatureColorScheme::entry_node:
                materialId = srcNode.entryNodeId;
                break;
            case VasculatureColorScheme::section_gradient:
                materialId = i * double(_nbMaxSegmentsPerSection) /
                             double(section.size());
                break;
            }

            const auto srcUserData = section[i];
            const Vector4f src = getBezierPoint(controlPoints, t);
            const auto sectionId = srcNode.sectionId;

            const auto srcRadius =
                (srcUserData < radii.size() ? radii[srcUserData] : src.w);

            if (!useSdf)
                geometryIndex = container.addSphere(Vector3f(src), srcRadius,
                                                    materialId, srcUserData);
            if (i > 0)
            {
                const auto dstUserData = section[i + 1];
                const auto& dstNode = _nodes[section[i + 1]];
                const Vector4f dst = getBezierPoint(controlPoints, t + step);
                const auto dstRadius =
                    (dstUserData < radii.size() ? radii[dstUserData] : dst.w);
                geometryIndex =
                    container.addCone(Vector3f(dst), dstRadius, Vector3f(src),
                                      srcRadius, materialId, srcUserData,
                                      neighbours,
                                      Vector3f(segmentDisplacementStrength,
                                               segmentDisplacementFrequency,
                                               0.f));
                neighbours.insert(geometryIndex);
            }
            ++i;
        }
#pragma omp critical
        containers.push_back(container);
    }

    for (size_t i = 0; i < containers.size(); ++i)
    {
        const float progress = 1.f + i;
        PLUGIN_PROGRESS("- Compiling 3D geometry...", progress,
                        containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }

    _applyPaletteToModel(model, details.palette);
}

void Vasculature::_buildEdges(Model& model)
{
    const auto& connector = DBConnector::getInstance();
    const auto edges = connector.getVasculatureEdges(_details.populationName,
                                                     _details.sqlFilter);

    const auto radiusMultiplier = _details.radiusMultiplier;

    ThreadSafeContainer container(model, _details.useSdf);
    uint64_t index = 0;
    for (const auto edge : edges)
    {
        PLUGIN_PROGRESS("Loading vasculature edges", index++, edges.size());
        size_t materialId = 0;
        if (_nodes.find(edge.first) == _nodes.end() ||
            _nodes.find(edge.second) == _nodes.end())
            continue;
        const auto& srcNode = _nodes[edge.first];
        const auto& srcPoint = srcNode.position;
        const auto srcRadius = srcNode.radius * radiusMultiplier;
        const auto& dstNode = _nodes[edge.second];
        const auto& dstPoint = dstNode.position;
        const auto dstRadius = dstNode.radius * radiusMultiplier;

        container.addCone(srcPoint, srcRadius, dstPoint, dstRadius, materialId,
                          0, {},
                          Vector3f(dstRadius * segmentDisplacementStrength,
                                   dstRadius * segmentDisplacementFrequency,
                                   0.f));
    }
    PLUGIN_INFO(1, "");

    container.commitToModel();
    PLUGIN_INFO(1, "");

    PLUGIN_ERROR("Created " + std::to_string(model.getMaterials().size()) +
                 " materials");
}

void Vasculature::_buildModel(const VasculatureColorSchemeDetails& details,
                              const doubles& radii)
{
    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto model = _scene.createModel();
    switch (_details.quality)
    {
    case VasculatureQuality::low:
        _buildGraphModel(*model, details);
        break;
    case VasculatureQuality::medium:
        _buildSimpleModel(*model, details, radii);
        break;
    default:
        _buildAdvancedModel(*model, details, radii);
        break;
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
    const floats radii =
        connector.getVasculatureSimulationTimeSeries(details.populationName,
                                                     details.simulationReportId,
                                                     details.frame);
    doubles series;
    for (const double radius : radii)
        series.push_back(details.amplitude * radius);
    _buildModel(VasculatureColorSchemeDetails(), series);
}

} // namespace vasculature
} // namespace bioexplorer