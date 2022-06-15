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

const uint64_t MAX_BATCH_SIZE =
    1000; // Max number of records returned by a batch query

Vasculature::Vasculature(Scene& scene, const VasculatureDetails& details)
    : SDFGeometries(details.radiusMultiplier == 0 ? 1.0
                                                  : details.radiusMultiplier,
                    doublesToVector3d(details.scale))
    , _details(details)
    , _scene(scene)
{
    Timer chrono;
    _buildModel();
    PLUGIN_TIMER(chrono.elapsed(), "Vasculature loaded");
}

void Vasculature::_addGraphSection(ThreadSafeContainer& container,
                                   const GeometryNode& srcNode,
                                   const GeometryNode& dstNode,
                                   const size_t materialId)
{
    const auto& src = srcNode.position;
    const auto& dst = dstNode.position;
    const auto userData = 0;
    const auto direction = dst - src;
    const auto maxRadius = std::max(srcNode.radius, dstNode.radius);
    const float radius = std::min(length(direction) / 5.0,
                                  maxRadius * _details.radiusMultiplier);
    container.addSphere(src, radius * 0.2, materialId, userData);
    container.addCone(src, radius * 0.2, Vector3f(src + direction * 0.79),
                      radius * 0.2, materialId, userData);
    container.addCone(dst, 0.0, Vector3f(src + direction * 0.8), radius,
                      materialId, userData);
    container.addCone(Vector3f(src + direction * 0.8), radius,
                      Vector3f(src + direction * 0.79), radius * 0.2,
                      materialId, userData);
}

void Vasculature::_addSimpleSection(ThreadSafeContainer& container,
                                    const GeometryNode& srcNode,
                                    const GeometryNode& dstNode,
                                    const size_t materialId)
{
    const auto& srcPoint = srcNode.position;
    const auto srcRadius = srcNode.radius * _details.radiusMultiplier;

    const auto& dstPoint = dstNode.position;
    const auto dstRadius = dstNode.radius * _details.radiusMultiplier;
    const auto userData = 0;

    if (!_details.useSdf)
    {
        container.addSphere(srcPoint, srcRadius, materialId, userData);
        container.addSphere(dstPoint, dstRadius, materialId, userData);
    }

    container.addCone(srcPoint, srcRadius, dstPoint, dstRadius, materialId,
                      userData, {},
                      Vector3f(segmentDisplacementStrength,
                               segmentDisplacementFrequency, 0.f));
}

void Vasculature::_addDetailedSection(ThreadSafeContainer& container,
                                      const GeometryNodes& nodes,
                                      const size_t baseMaterialId,
                                      const doubles& radii,
                                      const Vector2d& radiusRange)
{
    uint64_t geometryIndex = 0;
    Neighbours neighbours{};

    uint64_t i = 0;
    GeometryNode dstNode;
    for (const auto& node : nodes)
    {
        const auto& srcNode = node.second;
        const auto userData = node.first;

        size_t materialId = baseMaterialId;
        switch (_details.colorScheme)
        {
        case VasculatureColorScheme::radius:
            materialId = 256 * ((srcNode.radius - radiusRange.x) /
                                (radiusRange.y - radiusRange.x));
            break;
        case VasculatureColorScheme::section_points:
            materialId = 256 * double(node.first - nodes.begin()->first) /
                         double(nodes.size());
            break;
        }

        const auto& srcPoint = srcNode.position;
        const auto srcRadius =
            (userData < radii.size() ? radii[userData] : srcNode.radius) *
            _details.radiusMultiplier;
        const auto sectionId = srcNode.sectionId;

        if (i == 0 && !_details.useSdf)
            container.addSphere(srcPoint, srcRadius, materialId, userData);

        if (i > 0)
        {
            const auto dstPoint = dstNode.position;
            const double dstRadius =
                (userData < radii.size() ? radii[userData] : dstNode.radius) *
                _details.radiusMultiplier;

            if (!_details.useSdf)
                container.addSphere(dstPoint, dstRadius, materialId, userData);

            geometryIndex =
                container.addCone(dstPoint, dstRadius, srcPoint, srcRadius,
                                  materialId, userData, neighbours,
                                  Vector3f(segmentDisplacementStrength,
                                           segmentDisplacementFrequency, 0.f));
            neighbours = {geometryIndex};
        }

        dstNode = srcNode;
        ++i;
    }
}

void Vasculature::_addOrientation(ThreadSafeContainer& container,
                                  const GeometryNodes& nodes,
                                  const uint64_t sectionId)
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
            Vector4f(node.second.position,
                     node.second.radius * _details.radiusMultiplier));
        streamline.vertexColor.push_back(
            (i == 0 ? Vector4f(0.f, 0.f, 0.f, alpha)
                    : Vector4f(0.5 + 0.5 * normalize(node.second.position -
                                                     previousNode.position),
                               alpha)));
        previousNode = node.second;
        ++i;
    }

    container.addStreamline(sectionId, streamline);
}

void Vasculature::_buildModel(const doubles& radii)
{
    const auto useSdf =
        _details.representation == VasculatureRepresentation::graph
            ? false
            : _details.useSdf;

    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto model = _scene.createModel();
    ThreadSafeContainers containers;

    PLUGIN_INFO(1, "Identifying sections...");
    const auto nbDBConnections = DBConnector::getInstance().getNbConnections();

    const auto nbSections = DBConnector::getInstance().getVasculatureNbSections(
        _details.populationName, _details.sqlFilter);
    _nbSections = nbSections.x;
    const uint64_t nbMaxSections = nbSections.y;

    const auto nbQueries = nbMaxSections / MAX_BATCH_SIZE + 1;
    PLUGIN_INFO(1, "Number of sections="
                       << _nbSections << ", DB connections=" << nbDBConnections
                       << ", DB queries=" << nbQueries
                       << ", DB batch size=" << MAX_BATCH_SIZE);

    Vector2d radiusRange;
    if (_details.colorScheme == VasculatureColorScheme::radius)
        radiusRange = DBConnector::getInstance().getVasculatureRadiusRange(
            _details.populationName, _details.sqlFilter);

    uint64_t progress = 0;
    uint64_t index;
#pragma omp parallel for num_threads(nbDBConnections)
    for (index = 0; index < nbQueries; ++index)
    {
        const auto offset = index * MAX_BATCH_SIZE;
        const std::string limits = "section_guid>=" + std::to_string(offset) +
                                   " AND section_guid<" +
                                   std::to_string(offset + MAX_BATCH_SIZE);

        const auto filter = _details.sqlFilter;
        const auto nodes = DBConnector::getInstance().getVasculatureNodes(
            _details.populationName, filter, limits);

        if (nodes.empty())
            continue;

        ThreadSafeContainer container(*model, useSdf,
                                      doublesToVector3d(_details.scale));

        uint64_t nbLoadedSections = 0;
        auto iter = nodes.begin();
        do
        {
            GeometryNodes localNodes;
            const auto sectionId = iter->second.sectionId;
            auto previousSectionId = sectionId;
            while (iter != nodes.end() &&
                   iter->second.sectionId == previousSectionId)
            {
                localNodes[iter->first] = iter->second;
                ++iter;
            }

            const auto& srcNode = localNodes.begin()->second;
            auto it = localNodes.end();
            --it;
            const auto& dstNode = it->second;

            size_t materialId = 0;
            switch (_details.colorScheme)
            {
            case VasculatureColorScheme::section:
                materialId = sectionId;
                break;
            case VasculatureColorScheme::section_orientation:
                materialId = getMaterialIdFromOrientation(dstNode.position -
                                                          srcNode.position);
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
                materialId = 256 * ((srcNode.radius - radiusRange.x) /
                                    (radiusRange.y - radiusRange.x));
                break;
            }

            switch (_details.representation)
            {
            case VasculatureRepresentation::graph:
                _addGraphSection(container, srcNode, dstNode, materialId);
                break;
            case VasculatureRepresentation::section:
                _addSimpleSection(container, srcNode, dstNode, materialId);
                break;
            default:
                _addDetailedSection(container, localNodes, materialId, radii,
                                    radiusRange);
                break;
            }
            ++nbLoadedSections;
        } while (iter != nodes.end());

        if (omp_get_thread_num() == 0)
            PLUGIN_PROGRESS("Loading " << progress << "/" << _nbSections
                                       << " sections",
                            progress, _nbSections);

#pragma omp critical
        progress += nbLoadedSections;

#pragma omp critical
        containers.push_back(container);

#pragma omp critical
        _nbNodes += nodes.size();
    }

    for (size_t i = 0; i < containers.size(); ++i)
    {
        PLUGIN_PROGRESS("- Compiling 3D geometry...", 1 + i, containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }
    PLUGIN_INFO(1, "");

    const ModelMetadata metadata = {{"Number of nodes",
                                     std::to_string(_nbNodes)},
                                    {"Number of sections",
                                     std::to_string(_nbSections)}};

    _modelDescriptor.reset(new brayns::ModelDescriptor(std::move(model),
                                                       _details.assemblyName,
                                                       metadata));
    if (progress != _nbSections)
        PLUGIN_THROW("Vasculature model could not load all sections " +
                     std::to_string(progress) + " out of " +
                     std::to_string(_nbSections));

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
    _buildModel(series);
}

} // namespace vasculature
} // namespace bioexplorer
