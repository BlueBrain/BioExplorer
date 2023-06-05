/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "Vasculature.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

#include <science/io/db/DBConnector.h>

#include <platform/core/common/Timer.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>

namespace bioexplorer
{
namespace vasculature
{
using namespace common;
using namespace io;
using namespace db;

Vasculature::Vasculature(Scene& scene, const VasculatureDetails& details, const Vector3d& assemblyPosition,
                         const Quaterniond& assemblyRotation)
    : SDFGeometries(details.alignToGrid, assemblyPosition, assemblyRotation, doublesToVector3d(details.scale))
    , _details(details)
    , _scene(scene)
{
    _animationDetails = doublesToCellAnimationDetails(_details.animationParams);

    Timer chrono;
    _buildModel();
    PLUGIN_TIMER(chrono.elapsed(), "Vasculature loaded");
}

double Vasculature::_getDisplacementValue(const DisplacementElement& element)
{
    const auto params = _details.displacementParams;
    switch (element)
    {
    case DisplacementElement::vasculature_segment_strength:
        return valueFromDoubles(params, 0, DEFAULT_VASCULATURE_SEGMENT_STRENGTH);
    case DisplacementElement::vasculature_segment_frequency:
        return valueFromDoubles(params, 1, DEFAULT_VASCULATURE_SEGMENT_FREQUENCY);
    default:
        PLUGIN_THROW("Invalid displacement element");
    }
}

void Vasculature::_logRealismParams()
{
    PLUGIN_INFO(1, "----------------------------------------------------");
    PLUGIN_INFO(1, "Realism level (" << static_cast<uint32_t>(_details.realismLevel) << ")");
    PLUGIN_INFO(1,
                "- Section     : " << boolAsString(andCheck(static_cast<uint32_t>(_details.realismLevel),
                                                            static_cast<uint32_t>(VasculatureRealismLevel::section))));
    PLUGIN_INFO(1, "- Bifurcation : " << boolAsString(
                       andCheck(static_cast<uint32_t>(_details.realismLevel),
                                static_cast<uint32_t>(VasculatureRealismLevel::bifurcation))));
    PLUGIN_INFO(1, "----------------------------------------------------");
}

void Vasculature::_addGraphSection(ThreadSafeContainer& container, const GeometryNode& srcNode,
                                   const GeometryNode& dstNode, const size_t materialId)
{
    const auto userData = NO_USER_DATA;
    const auto useSdf = false;
    const auto maxRadius = std::max(srcNode.radius, dstNode.radius);
    const auto src = _animatedPosition(Vector4d(srcNode.position, maxRadius));
    const auto dst = _animatedPosition(Vector4d(dstNode.position, maxRadius));
    const auto direction = dst - src;

    const float radius = std::min(length(direction) / 5.0, _getCorrectedRadius(maxRadius, _details.radiusMultiplier));
    container.addSphere(src, radius * 0.2, materialId, useSdf, userData);
    container.addCone(src, radius * 0.2, Vector3f(src + direction * 0.79), radius * 0.2, materialId, useSdf, userData);
    container.addCone(dst, 0.0, Vector3f(src + direction * 0.8), radius, materialId, useSdf, userData);
    container.addCone(Vector3f(src + direction * 0.8), radius, Vector3f(src + direction * 0.79), radius * 0.2,
                      materialId, useSdf, userData);
}

void Vasculature::_addSimpleSection(ThreadSafeContainer& container, const GeometryNode& srcNode,
                                    const GeometryNode& dstNode, const size_t materialId, const uint64_t userData)
{
    const auto srcRadius = _getCorrectedRadius(srcNode.radius, _details.radiusMultiplier);
    const auto& srcPoint = _animatedPosition(Vector4d(srcNode.position, srcRadius));

    const auto dstRadius = _getCorrectedRadius(dstNode.radius, _details.radiusMultiplier);
    const auto& dstPoint = _animatedPosition(Vector4d(dstNode.position, dstRadius));

    const auto useSdf =
        andCheck(static_cast<uint32_t>(_details.realismLevel), static_cast<uint32_t>(VasculatureRealismLevel::section));
    if (!useSdf)
    {
        container.addSphere(srcPoint, srcRadius, materialId, useSdf, userData);
        container.addSphere(dstPoint, dstRadius, materialId, useSdf, userData);
    }

    container.addCone(srcPoint, srcRadius, dstPoint, dstRadius, materialId, useSdf, userData, {},
                      Vector3f(_getDisplacementValue(DisplacementElement::vasculature_segment_strength),
                               _getDisplacementValue(DisplacementElement::vasculature_segment_frequency), 0.f));
}

void Vasculature::_addDetailedSection(ThreadSafeContainer& container, const GeometryNodes& nodes,
                                      const size_t baseMaterialId, const doubles& radii, const Vector2d& radiusRange)
{
    uint64_t geometryIndex = 0;
    Neighbours neighbours;

    const auto useSdf =
        andCheck(static_cast<uint32_t>(_details.realismLevel), static_cast<uint32_t>(VasculatureRealismLevel::section));

    GeometryNodes localNodes;
    switch (_details.representation)
    {
    case VasculatureRepresentation::optimized_segment:
    {
        double oldRadius = 0.0;
        double segmentLength = 0.0;
        for (const auto& node : nodes)
        {
            segmentLength += node.second.radius;
            if (segmentLength < 2.0 * (oldRadius + node.second.radius))
            {
                GeometryNode n;
                n.position = node.second.position;
                n.radius = node.second.radius;
                localNodes[node.first] = n;
            }
            else
                segmentLength = 0.0;
            oldRadius = node.second.radius;
        }
        break;
    }
    case VasculatureRepresentation::bezier:
    {
        Vector4fs points;
        uint64_ts ids;
        for (const auto& node : nodes)
        {
            points.push_back(
                Vector4d(node.second.position.x, node.second.position.y, node.second.position.z, node.second.radius));
            ids.push_back(node.first);
        }
        const auto localPoints = _getProcessedSectionPoints(MorphologyRepresentation::bezier, points);

        uint64_t i = 0;
        for (const auto& point : localPoints)
        {
            GeometryNode n;
            n.position = Vector3d(point.x, point.y, point.y);
            n.radius = point.w;
            localNodes[ids[i * (nodes.size() / localPoints.size())]] = n;
            ++i;
        }
    }
    default:
        localNodes = nodes;
    }

    uint64_t i = 0;
    GeometryNode dstNode;
    for (const auto& node : localNodes)
    {
        const auto& srcNode = node.second;
        const auto userData = node.first;

        size_t materialId;
        switch (_details.colorScheme)
        {
        case VasculatureColorScheme::radius:
            materialId = 256 * ((srcNode.radius - radiusRange.x) / (radiusRange.y - radiusRange.x));
            break;
        case VasculatureColorScheme::section_points:
            materialId = 256 * double(node.first - nodes.begin()->first) / double(nodes.size());
            break;
        default:
            materialId = baseMaterialId;
            break;
        }

        const float srcRadius = _getCorrectedRadius((userData < radii.size() ? radii[userData] : srcNode.radius),
                                                    _details.radiusMultiplier);
        const auto srcPosition = _animatedPosition(Vector4d(srcNode.position, srcRadius));

        if (i == 0)
            container.addSphere(srcPosition, srcRadius, materialId, useSdf, userData);
        else
        {
            const float dstRadius = _getCorrectedRadius((userData < radii.size() ? radii[userData] : dstNode.radius),
                                                        _details.radiusMultiplier);
            const auto dstPosition = _animatedPosition(Vector4d(dstNode.position, dstRadius));

            geometryIndex =
                container.addCone(srcPosition, srcRadius, dstPosition, dstRadius, materialId, useSdf, userData,
                                  neighbours,
                                  Vector3f(_getDisplacementValue(DisplacementElement::vasculature_segment_strength),
                                           _getDisplacementValue(DisplacementElement::vasculature_segment_frequency),
                                           0.f));
            neighbours = {geometryIndex};

            if (!useSdf)
                neighbours.insert(container.addSphere(srcPosition, srcRadius, materialId, useSdf, userData));
        }

        dstNode = srcNode;
        ++i;
    }
}

void Vasculature::_addOrientation(ThreadSafeContainer& container, const GeometryNodes& nodes, const uint64_t sectionId)
{
    const auto nbNodes = nodes.size();
    if (nbNodes <= 3)
        return;

    StreamlinesData streamline;

    GeometryNode previousNode;
    const float alpha = 1.f;
    uint64_t i = 0;
    for (const auto& node : nodes)
    {
        streamline.vertex.push_back(
            Vector4f(node.second.position, _getCorrectedRadius(node.second.radius, _details.radiusMultiplier)));
        streamline.vertexColor.push_back(
            (i == 0 ? Vector4f(0.f, 0.f, 0.f, alpha)
                    : Vector4f(0.5 + 0.5 * normalize(node.second.position - previousNode.position), alpha)));
        previousNode = node.second;
        ++i;
    }

    container.addStreamline(sectionId, streamline);
}

void Vasculature::_buildModel(const doubles& radii)
{
    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto model = _scene.createModel();
    ThreadSafeContainers containers;

    PLUGIN_INFO(1, "Identifying nodes...");
    const auto nbDBConnections = DBConnector::getInstance().getNbConnections();

    _nbNodes = DBConnector::getInstance().getVasculatureNbNodes(_details.populationName, _details.sqlFilter);

    const auto dbBatchSize = _nbNodes / nbDBConnections;
    PLUGIN_INFO(1, "DB connections=" << nbDBConnections << ", DB batch size=" << dbBatchSize);

    Vector2d radiusRange;
    if (_details.colorScheme == VasculatureColorScheme::radius)
        radiusRange = DBConnector::getInstance().getVasculatureRadiusRange(_details.populationName, _details.sqlFilter);

    uint64_t progress = 0;
    uint64_t index;
#pragma omp parallel for num_threads(nbDBConnections)
    for (index = 0; index < nbDBConnections; ++index)
    {
        const auto offset = index * dbBatchSize;
        const std::string limits = "OFFSET " + std::to_string(offset) + " LIMIT " + std::to_string(dbBatchSize);

        const auto filter = _details.sqlFilter;
        const auto nodes = DBConnector::getInstance().getVasculatureNodes(_details.populationName, filter, limits);

        if (nodes.empty())
            continue;

        ThreadSafeContainer container(*model, _alignToGrid, _position, _rotation, doublesToVector3d(_details.scale));

        auto iter = nodes.begin();
        uint64_t previousSectionId = iter->second.sectionId;
        do
        {
            GeometryNodes sectionNodes;
            const auto sectionId = iter->second.sectionId;
            const auto userData = iter->first;
            while (iter != nodes.end() && iter->second.sectionId == previousSectionId)
            {
                sectionNodes[iter->first] = iter->second;
                ++iter;
            }
            previousSectionId = sectionId;

            if (sectionNodes.size() >= 1)
            {
                const auto& srcNode = sectionNodes.begin()->second;
                auto it = sectionNodes.end();
                --it;
                const auto& dstNode = it->second;

                size_t materialId;
                switch (_details.colorScheme)
                {
                case VasculatureColorScheme::section:
                    materialId = sectionId;
                    break;
                case VasculatureColorScheme::section_orientation:
                    materialId = getMaterialIdFromOrientation(dstNode.position - srcNode.position);
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
                case VasculatureColorScheme::radius:
                    materialId = 256 * ((srcNode.radius - radiusRange.x) / (radiusRange.y - radiusRange.x));
                    break;
                case VasculatureColorScheme::region:
                    materialId = dstNode.regionId;
                    break;
                default:
                    materialId = 0;
                    break;
                }

                switch (_details.representation)
                {
                case VasculatureRepresentation::graph:
                    _addGraphSection(container, srcNode, dstNode, materialId);
                    break;
                case VasculatureRepresentation::section:
                    _addSimpleSection(container, srcNode, dstNode, materialId, userData);
                    break;
                default:
                    _addDetailedSection(container, sectionNodes, materialId, radii, radiusRange);
                    break;
                }
            }
        } while (iter != nodes.end());

        PLUGIN_PROGRESS("Loading nodes", progress, nbDBConnections);

#pragma omp critical
        ++progress;

#pragma omp critical
        containers.push_back(container);
    }

    for (size_t i = 0; i < containers.size(); ++i)
    {
        PLUGIN_PROGRESS("- Compiling 3D geometry...", 1 + i, containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }
    PLUGIN_INFO(1, "");

    const ModelMetadata metadata = {{"Number of nodes", std::to_string(_nbNodes)}, {"SQL filter", _details.sqlFilter}};

    _modelDescriptor.reset(new core::ModelDescriptor(std::move(model), _details.assemblyName, metadata));

    if (_modelDescriptor)
        _scene.addModel(_modelDescriptor);
    else
        PLUGIN_THROW(
            "Vasculature model could not be created for "
            "population " +
            _details.populationName);
}

void Vasculature::setRadiusReport(const VasculatureRadiusReportDetails& details)
{
    auto& connector = DBConnector::getInstance();
    const auto simulationReport = connector.getSimulationReport(details.populationName, details.simulationReportId);

    const size_t nbFrames = (simulationReport.endTime - simulationReport.startTime) / simulationReport.timeStep;
    if (nbFrames == 0)
        PLUGIN_THROW("Report does not contain any simulation data: " + simulationReport.description);

    if (details.frame >= nbFrames)
        PLUGIN_THROW("Invalid frame specified for report: " + simulationReport.description);
    const floats radii =
        connector.getVasculatureSimulationTimeSeries(details.populationName, details.simulationReportId, details.frame);
    doubles series;
    for (const double radius : radii)
        series.push_back(details.amplitude * radius);
    _buildModel(series);
}

} // namespace vasculature
} // namespace bioexplorer
