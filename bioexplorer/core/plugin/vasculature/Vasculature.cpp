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

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/parameters/ParametersManager.h>

namespace bioexplorer
{
namespace vasculature
{
using namespace common;
using namespace geometry;
using namespace io;
using namespace db;

Vasculature::Vasculature(Scene& scene, const VasculatureDetails& details)
    : _details(details)
    , _scene(scene)
{
    _importFromDB();
}

// From http://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
    almost_equal(T x, T y, int ulp)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x - y) <=
               std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
           // unless the result is subnormal
           || std::abs(x - y) < std::numeric_limits<T>::min();
}

// TODO: Generalise SDF for any type of asset
size_t Vasculature::_addSDFGeometry(SDFMorphologyData& sdfMorphologyData,
                                    const SDFGeometry& geometry,
                                    const std::set<size_t>& neighbours,
                                    const size_t materialId, const int section)
{
    const size_t idx = sdfMorphologyData.geometries.size();
    sdfMorphologyData.geometries.push_back(geometry);
    sdfMorphologyData.neighbours.push_back(neighbours);
    sdfMorphologyData.materials.push_back(materialId);
    sdfMorphologyData.geometrySection[idx] = section;
    sdfMorphologyData.sectionGeometries[section].push_back(idx);
    return idx;
}

void Vasculature::_addStepConeGeometry(
    const bool useSDF, const Vector3d& position, const double radius,
    const Vector3d& target, const double previousRadius,
    const size_t materialId, const uint64_t& userDataOffset, Model& model,
    SDFMorphologyData& sdfMorphologyData, const uint32_t sdfGroupId,
    const Vector3f& displacementParams)
{
    if (useSDF)
    {
        const auto geom =
            (almost_equal(radius, previousRadius, 100000))
                ? createSDFPill(position, target, radius, userDataOffset,
                                displacementParams)
                : createSDFConePill(position, target, radius, previousRadius,
                                    userDataOffset, displacementParams);
        _addSDFGeometry(sdfMorphologyData, geom, {}, materialId, sdfGroupId);
    }
    else if (almost_equal(radius, previousRadius, 100000))
        model.addCylinder(materialId,
                          {position, target, static_cast<float>(radius),
                           userDataOffset});
    else
        model.addCone(materialId,
                      {position, target, static_cast<float>(radius),
                       static_cast<float>(previousRadius), userDataOffset});
}

void Vasculature::_finalizeSDFGeometries(Model& model,
                                         SDFMorphologyData& sdfMorphologyData)
{
    const size_t numGeoms = sdfMorphologyData.geometries.size();
    sdfMorphologyData.localToGlobalIdx.resize(numGeoms, 0);

    // Extend neighbours to make sure smoothing is applied on all
    // closely connected geometries
    for (size_t rep = 0; rep < 4; rep++)
    {
        const size_t numNeighs = sdfMorphologyData.neighbours.size();
        auto neighsCopy = sdfMorphologyData.neighbours;
        for (size_t i = 0; i < numNeighs; i++)
        {
            for (size_t j : sdfMorphologyData.neighbours[i])
            {
                for (size_t newNei : sdfMorphologyData.neighbours[j])
                {
                    neighsCopy[i].insert(newNei);
                    neighsCopy[newNei].insert(i);
                }
            }
        }
        sdfMorphologyData.neighbours = neighsCopy;
    }

    for (size_t i = 0; i < numGeoms; i++)
    {
        // Convert neighbours from set to vector and erase itself from its
        // neighbours
        std::vector<size_t> neighbours;
        const auto& neighSet = sdfMorphologyData.neighbours[i];
        std::copy(neighSet.begin(), neighSet.end(),
                  std::back_inserter(neighbours));
        neighbours.erase(std::remove_if(neighbours.begin(), neighbours.end(),
                                        [i](size_t elem) { return elem == i; }),
                         neighbours.end());

        model.addSDFGeometry(sdfMorphologyData.materials[i],
                             sdfMorphologyData.geometries[i], neighbours);
    }
}

void Vasculature::_importFromDB()
{
    auto& connector = DBConnector::getInstance();

    std::string sqlFilter = _details.sqlFilter;
    if (!_details.loadCapilarities)
    {
        if (!sqlFilter.empty())
            sqlFilter += " AND ";
        sqlFilter += "v.type_guid != 1";
    }

    _nodes = connector.getVasculatureNodes(_details.populationName, sqlFilter);

    PLUGIN_INFO(1, "Full vasculature is made of " << _nodes.size() << " nodes");

    std::map<uint64_t, std::vector<uint64_t>> pairs;
    std::set<uint64_t> entryNodes;
    const auto& gids = _details.gids;
    for (const auto& node : _nodes)
    {
        if (!_details.loadCapilarities &&
            node.second.type == static_cast<uint64_t>(EdgeType::capilarity))
            // Ignore capilarities
            continue;

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
        if (pair.second.size() == 2)
        {
            auto& edge1 = _nodes[pair.second[0]];
            edge1.pairId = pair.second[1];
            auto& edge2 = _nodes[pair.second[1]];
            edge2.pairId = pair.second[1];
            const auto graphId1 = edge1.graphId;
            const auto graphId2 = edge2.graphId;
            for (auto& node : _nodes)
                if (node.second.graphId == graphId1 ||
                    node.second.graphId == graphId2)
                    node.second.pairId = pair.first;
        }
        else
            PLUGIN_ERROR("Invalid number of Ids for pair " << pair.first);
    }

    _nbPairs = pairs.size();
    _nbEntryNodes = entryNodes.size();
    PLUGIN_INFO(1,
                "Loaded vasculature is made of " << _nodes.size() << " nodes");

    _buildModel();
}

void Vasculature::_buildModel(const VasculatureColorSchemeDetails& details)
{
    auto model = _scene.createModel();
    std::set<uint64_t> materialIds;
    SDFMorphologyData sdfMorphologyData;
    const auto useSdf = _details.useSdf;

    uint64_t edgeCount = 0;
    for (const auto& section : _sections)
    {
        bool firstControlPoint = true;
        Vector3d dst;
        double dstRadius = 0.0;

        Vector4ds controlPoints;
        const uint64_t nbControlPoints = section.second.size();
        for (const auto& nodeId : section.second)
        {
            const auto& node = _nodes[nodeId];
            controlPoints.push_back({node.position.x, node.position.y,
                                     node.position.z, node.radius});
        }

        double tStep;
        switch (_details.quality)
        {
        case VasculatureQuality::low:
            tStep = 0.99;
            break;
        case VasculatureQuality::medium:
            tStep = 5.0 / double(nbControlPoints);
            break;
        default:
            tStep = 1.0 / double(nbControlPoints);
            break;
        }

        for (double t = 0.0; t < 1.0; t += tStep)
        {
            const uint64_t nodeId = section.second[t * double(nbControlPoints)];
            const auto& node = _nodes[nodeId];
            uint64_t materialId = 0;

            switch (details.colorScheme)
            {
            case VasculatureColorScheme::edge:
                materialId = edgeCount;
                break;
            case VasculatureColorScheme::section:
                materialId = node.sectionId;
                break;
            case VasculatureColorScheme::subgraph:
                materialId = node.graphId;
                break;
            case VasculatureColorScheme::pair:
                materialId = node.pairId;
                break;
            case VasculatureColorScheme::entry_node:
                materialId = node.entryNodeId;
                break;
            default:
                break;
            }
            materialIds.insert(materialId);

            const Vector4d src4d = getBezierPoint(controlPoints, t);
            const Vector3d src{src4d.x, src4d.y, src4d.z};
            const double srcRadius = src4d.w * _details.radiusMultiplier;

            _bounds.merge(src);

            if (!firstControlPoint)
            {
                const uint64_t userData = nodeId;
                if (useSdf)
                    _addSDFGeometry(sdfMorphologyData,
                                    createSDFSphere(src, srcRadius, userData),
                                    {}, materialId, node.sectionId);
                else
                    model->addSphere(materialId,
                                     {src, static_cast<float>(srcRadius),
                                      userData});

                _addStepConeGeometry(useSdf, src, srcRadius, dst, dstRadius,
                                     materialId, userData, *model,
                                     sdfMorphologyData, node.sectionId);
            }
            dst = src;
            dstRadius = srcRadius;
            firstControlPoint = false;

            ++edgeCount;
        }
    }

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

    if (useSdf)
        _finalizeSDFGeometries(*model, sdfMorphologyData);

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
