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
    _buildModel();
    PLUGIN_TIMER(chrono.elapsed(), "Vasculature loaded");
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

    auto& connector = DBConnector::getInstance();

    PLUGIN_INFO(1, "Loading sections...");
    const auto sections =
        connector.getVasculatureSections(_details.populationName,
                                         _details.sqlFilter);

    ThreadSafeContainers containers;
    uint64_t counter = 0;
    uint64_t index;
#pragma omp parallel for
    for (index = 0; index < sections.size(); ++index)
    {
        const auto sectionId = sections[index];

        const auto filter =
            "section_guid=" + std::to_string(sectionId) +
            (_details.sqlFilter.empty() ? "" : " AND " + _details.sqlFilter);
        const auto nodes =
            connector.getVasculatureNodes(_details.populationName, filter);

        if (nodes.size() < 2)
            continue;

        ThreadSafeContainer container(model, _details.useSdf,
                                      doublesToVector3d(_details.scale));

        const auto& srcNode = nodes.begin()->second;
        const auto& dstNode = (--nodes.end())->second;

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
        const auto userData = nodes.begin()->first;
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

#pragma omp critical
        ++counter;

#pragma omp critical
        _nbNodes += nodes.size();

#pragma omp critical
        PLUGIN_PROGRESS("Loading " << sections.size() << " sections", counter,
                        sections.size());
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

    auto& connector = DBConnector::getInstance();

    PLUGIN_INFO(1, "Loading sections...");
    const auto sections =
        connector.getVasculatureSections(_details.populationName,
                                         _details.sqlFilter);

    ThreadSafeContainers containers;
    uint64_t counter = 0;
    uint64_t index;
#pragma omp parallel
    for (index = 0; index < sections.size(); ++index)
    {
        const auto sectionId = sections[index];

        const auto filter =
            "section_guid=" + std::to_string(sectionId) +
            (_details.sqlFilter.empty() ? "" : " AND " + _details.sqlFilter);
        const auto nodes =
            connector.getVasculatureNodes(_details.populationName, filter);

        if (nodes.size() < 2)
            continue;

        ThreadSafeContainer container(model, _details.useSdf,
                                      doublesToVector3d(_details.scale));

        uint64_t geometryIndex = 0;
        size_t materialId = 0;
        const auto srcNode = nodes.begin();
        const auto userData = srcNode->first;

        switch (details.colorScheme)
        {
        case VasculatureColorScheme::section:
            materialId = srcNode->second.sectionId;
            break;
        case VasculatureColorScheme::subgraph:
            materialId = srcNode->second.graphId;
            break;
        case VasculatureColorScheme::pair:
            materialId = srcNode->second.pairId;
            break;
        case VasculatureColorScheme::entry_node:
            materialId = srcNode->second.entryNodeId;
            break;
        }

        const auto& srcPoint = srcNode->second.position;
        const auto srcRadius =
            (userData < radii.size() ? radii[userData]
                                     : srcNode->second.radius) *
            radiusMultiplier;

        const auto& dstNode = (--nodes.end())->second;
        const auto& dstPoint = dstNode.position;
        const auto dstRadius =
            (userData < radii.size() ? radii[userData] : dstNode.radius) *
            radiusMultiplier;

        if (!useSdf)
            container.addSphere(dstPoint, dstRadius, materialId, userData);

        container.addCone(dstPoint, dstRadius, srcPoint, srcRadius, materialId,
                          userData, {},
                          Vector3f(segmentDisplacementStrength,
                                   segmentDisplacementFrequency, 0.f));
#pragma omp critical
        containers.push_back(container);

#pragma omp critical
        ++counter;

#pragma omp critical
        _nbNodes += nodes.size();

#pragma omp critical
        PLUGIN_PROGRESS("Loading " << sections.size() << " sections", counter,
                        sections.size());
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

    auto& connector = DBConnector::getInstance();

    PLUGIN_INFO(1, "Loading sections...");
    const auto sections =
        connector.getVasculatureSections(_details.populationName,
                                         _details.sqlFilter);

    ThreadSafeContainers containers;
    uint64_t counter = 0;
    uint64_t index;
#pragma omp parallel
    for (index = 0; index < sections.size(); ++index)
    {
        const auto sectionId = sections[index];

        const auto filter =
            "section_guid=" + std::to_string(sectionId) +
            (_details.sqlFilter.empty() ? "" : " AND " + _details.sqlFilter);
        const auto nodes =
            connector.getVasculatureNodes(_details.populationName, filter);

        if (nodes.size() < 2)
            continue;

        ThreadSafeContainer container(model, _details.useSdf,
                                      doublesToVector3d(_details.scale));

        uint64_t geometryIndex = 0;
        size_t materialId = 0;
        size_t previousMaterialId = 0;
        Neighbours neighbours{};

        uint64_t i = 0;
        GeometryNode dstNode;

        for (const auto& node : nodes)
        {
            const auto& srcNode = node.second;
            const auto userData = node.first;

            switch (details.colorScheme)
            {
            case VasculatureColorScheme::section:
                materialId = srcNode.sectionId;
                break;
            case VasculatureColorScheme::subgraph:
                materialId = srcNode.graphId;
                break;
            case VasculatureColorScheme::node:
                materialId = userData;
                break;
            case VasculatureColorScheme::pair:
                materialId = srcNode.pairId;
                break;
            case VasculatureColorScheme::entry_node:
                materialId = srcNode.entryNodeId;
                break;
            case VasculatureColorScheme::section_gradient:
                materialId = nodes.size() * (double(i) / double(nodes.size()));
                break;
            }

            const auto& srcPoint = srcNode.position;
            const auto srcRadius =
                (userData < radii.size() ? radii[userData] : srcNode.radius) *
                radiusMultiplier;
            const auto sectionId = srcNode.sectionId;

            dstNode = srcNode;

            if (i == 0 && !useSdf)
            {
                container.addSphere(srcPoint, srcRadius, materialId, userData);
                continue;
            }

            const Vector3d dstPoint = dstNode.position;
            const double dstRadius =
                (userData < radii.size() ? radii[userData] : dstNode.radius) *
                radiusMultiplier;

            if (!useSdf)
                container.addSphere(dstPoint, dstRadius, materialId, userData);

            if (i < nodes.size() - 1)
                geometryIndex =
                    container.addCone(dstPoint, dstRadius, srcPoint, srcRadius,
                                      previousMaterialId, userData, neighbours,
                                      Vector3f(segmentDisplacementStrength,
                                               segmentDisplacementFrequency,
                                               0.f));
            neighbours = {geometryIndex};
            ++i;
        }
#pragma omp critical
        containers.push_back(container);

#pragma omp critical
        ++counter;

#pragma omp critical
        _nbNodes += nodes.size();

#pragma omp critical
        PLUGIN_PROGRESS("Loading " << sections.size() << " sections", counter,
                        sections.size());
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

    switch (details.colorScheme)
    {
    case VasculatureColorScheme::subgraph:
        _nbGraphs = model->getMaterials().size();
        break;
    case VasculatureColorScheme::pair:
        _nbPairs = model->getMaterials().size();
        break;
    case VasculatureColorScheme::entry_node:
        _nbEntryNodes = model->getMaterials().size();
        break;
    }

    ModelMetadata metadata = {{"Number of nodes", std::to_string(_nbNodes)},
                              {"Number of sections",
                               std::to_string(_nbSections)},
                              {"Number of sub-graphs",
                               std::to_string(_nbGraphs)}};

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
