/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include "Astrocytes.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>
#include <plugin/common/ThreadSafeContainer.h>
#include <plugin/common/Utils.h>

#include <plugin/io/db/DBConnector.h>

#include <brayns/common/Timer.h>
#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
namespace morphology
{
using namespace common;
using namespace io;
using namespace db;

const double DEFAULT_MITOCHONDRIA_DENSITY = 0.0459;
const double DEFAULT_ENDFOOT_RADIUS_RATIO = 1.5;

Astrocytes::Astrocytes(Scene& scene, const AstrocytesDetails& details)
    : Morphologies(details.radiusMultiplier, doublesToVector3d(details.scale))
    , _details(details)
    , _scene(scene)
{
    _radiusMultiplier =
        _details.radiusMultiplier > 0.0 ? _details.radiusMultiplier : 1.0;
    _animationDetails = doublesToCellAnimationDetails(_details.animationParams);
    Timer chrono;
    _buildModel();
    PLUGIN_TIMER(chrono.elapsed(), "Astrocytes loaded");
}

void Astrocytes::_buildModel(const doubles& radii)
{
    const auto animationParams =
        doublesToMolecularSystemAnimationDetails(_details.animationParams);
    srand(animationParams.seed);

    if (_modelDescriptor)
        _scene.removeModel(_modelDescriptor->getModelID());

    auto& connector = DBConnector::getInstance();

    auto model = _scene.createModel();
    const auto useSdf = _details.useSdf;
    const auto somas = connector.getAstrocytes(_details.sqlFilter);
    const auto loadEndFeet = !_details.vasculaturePopulationName.empty();

    PLUGIN_INFO(1, "Building " << somas.size() << " astrocytes");

    // Astrocytes
    size_t baseMaterialId = 0;
    const uint64_t userData = 0;

    ThreadSafeContainers containers;
    const auto nbDBConnections = DBConnector::getInstance().getNbConnections();
    uint64_t index;
#pragma omp parallel for num_threads(nbDBConnections)
    for (index = 0; index < somas.size(); ++index)
    {
        if (omp_get_thread_num() == 0)
            PLUGIN_PROGRESS("Loading astrocytes", index,
                            somas.size() / nbDBConnections);

        auto it = somas.begin();
        std::advance(it, index);
        const auto& soma = it->second;
        const auto somaId = it->first;

        ThreadSafeContainer container(*model, useSdf, _scale);

        // Load data from DB
        double somaRadius = 0.0;
        SectionMap sections;
        if (_details.loadSomas || _details.loadDendrites)
            sections = connector.getAstrocyteSections(somaId);

        EndFootMap endFeet;
        if (loadEndFeet)
            endFeet = connector.getAstrocyteEndFeet(
                _details.vasculaturePopulationName, somaId);

        // Soma radius
        uint64_t count = 1;
        for (const auto& section : sections)
            if (section.second.parentId == SOMA_AS_PARENT)
            {
                const auto& point = section.second.points[0];
                somaRadius += 0.75 * length(Vector3d(point));
                ++count;
            }
        somaRadius = _radiusMultiplier * somaRadius / count;
        const auto somaPosition =
            _animatedPosition(Vector4d(soma.center, somaRadius), somaId);

        // Color scheme
        switch (_details.populationColorScheme)
        {
        case PopulationColorScheme::id:
            baseMaterialId = somaId * NB_MATERIALS_PER_MORPHOLOGY;
            break;
        default:
            baseMaterialId = 0;
        }

        const auto somaMaterialId =
            baseMaterialId + (_details.morphologyColorScheme ==
                                      MorphologyColorScheme::section_type
                                  ? MATERIAL_OFFSET_SOMA
                                  : 0);

        uint64_t somaGeometryIndex = 0;
        if (_details.loadSomas)
        {
            somaGeometryIndex = container.addSphere(
                somaPosition, somaRadius, somaMaterialId, NO_USER_DATA, {},
                Vector3f(somaRadius * astrocyteSomaDisplacementStrength,
                         somaRadius * astrocyteSomaDisplacementFrequency, 0.f));
            if (_details.generateInternals)
                _addSomaInternals(container, baseMaterialId, somaPosition,
                                  somaRadius, DEFAULT_MITOCHONDRIA_DENSITY);
        }

        Neighbours neighbours;
        neighbours.insert(somaGeometryIndex);
        for (const auto& section : sections)
        {
            uint64_t geometryIndex = 0;
            const auto& points = section.second.points;

            size_t sectionMaterialId = baseMaterialId;
            const auto sectionId = section.first;
            switch (_details.morphologyColorScheme)
            {
            case MorphologyColorScheme::section_type:
                sectionMaterialId = baseMaterialId + section.second.type;
                break;
            case MorphologyColorScheme::section_orientation:
            {
                sectionMaterialId = getMaterialIdFromOrientation(
                    Vector3d(points[points.size() - 1]) - Vector3d(points[0]));
                break;
            }
            default:
                break;
            }
            size_t step = 1;
            switch (_details.morphologyRepresentation)
            {
            case MorphologyRepresentation::section:
                step = points.size() - 2;
                break;
            default:
                break;
            }

            if (_details.loadDendrites)
            {
                uint64_t geometryIndex = 0;
                if (section.second.parentId == SOMA_AS_PARENT)
                {
                    // Section connected to the soma
                    const auto& point = points[0];
                    const float srcRadius =
                        somaRadius * 0.75f * _radiusMultiplier;
                    const float dstRadius = point.w * 0.5f * _radiusMultiplier;
                    const auto dstPosition = _animatedPosition(
                        Vector4d(somaPosition + Vector3d(point), dstRadius),
                        somaId);
                    geometryIndex = container.addCone(
                        somaPosition, srcRadius, dstPosition, dstRadius,
                        somaMaterialId, userData, neighbours,
                        Vector3f(srcRadius * astrocyteSomaDisplacementStrength,
                                 srcRadius * astrocyteSomaDisplacementFrequency,
                                 0.f));
                    neighbours.insert(geometryIndex);
                }

                for (uint64_t i = 0; i < points.size() - 1; i += step)
                {
                    const auto srcPoint = points[i];
                    const float srcRadius =
                        srcPoint.w * 0.5 * _radiusMultiplier;
                    const auto src = _animatedPosition(
                        Vector4d(somaPosition + Vector3d(srcPoint), srcRadius),
                        somaId);

                    // Ignore points that are too close the previous one
                    // (according to respective radii)
                    Vector4f dstPoint;
                    float dstRadius;
                    do
                    {
                        dstPoint = points[i + step];
                        dstRadius = dstPoint.w * 0.5 * _radiusMultiplier;
                        ++i;
                    } while (length(Vector3f(dstPoint) - Vector3f(srcPoint)) <
                                 (srcRadius + dstRadius) &&
                             (i + step) < points.size() - 1);
                    --i;

                    const auto dst = _animatedPosition(
                        Vector4d(somaPosition + Vector3d(dstPoint), dstRadius),
                        somaId);
                    if (!useSdf)
                        geometryIndex = container.addSphere(dst, dstRadius,
                                                            sectionMaterialId,
                                                            NO_USER_DATA);

                    geometryIndex = container.addCone(
                        src, srcRadius, dst, dstRadius, sectionMaterialId,
                        userData, {geometryIndex},
                        Vector3f(srcRadius * sectionDisplacementStrength,
                                 sectionDisplacementFrequency, 0.f));

                    _bounds.merge(srcPoint);
                }
            }
        }

        if (loadEndFeet)
            _addEndFoot(container, endFeet, radii, somaMaterialId);
#pragma omp critical
        containers.push_back(container);
    }

    for (uint64_t i = 0; i < containers.size(); ++i)
    {
        PLUGIN_PROGRESS("- Compiling 3D geometry...", i, containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }

    ModelMetadata metadata = {
        {"Number of astrocytes", std::to_string(somas.size())}};

    _modelDescriptor.reset(new brayns::ModelDescriptor(std::move(model),
                                                       _details.assemblyName,
                                                       metadata));
    if (_modelDescriptor)
        _scene.addModel(_modelDescriptor);
    else
        PLUGIN_THROW("Astrocytes model could not be created");
}

void Astrocytes::_addEndFoot(ThreadSafeContainer& container,
                             const EndFootMap& endFeet, const doubles& radii,
                             const size_t materialId)
{
    const auto radiusMultiplier = _details.radiusMultiplier;
    const Vector3f displacement{sectionDisplacementStrength,
                                sectionDisplacementFrequency, 0.f};

    for (const auto& endFoot : endFeet)
    {
        for (const auto& node : endFoot.second.nodes)
        {
            const auto& connector = DBConnector::getInstance();
            const auto vasculatureNodes = connector.getVasculatureNodes(
                _details.vasculaturePopulationName,
                "section_guid=" +
                    std::to_string(endFoot.second.vasculatureSectionId));

            uint64_t startIndex = 0;
            uint64_t endIndex = 1;
            const auto halfLength = endFoot.second.length / 2.0;
            auto it = vasculatureNodes.begin();
            std::advance(it, endFoot.second.vasculatureSegmentId);
            const auto centerPosition = it->second.position;

            double length = 0.0;
            int64_t i = -1;
            // Find start segment making the assumption that the segment Id is
            // in the middle of the end-foot
            while (length < halfLength &&
                   endFoot.second.vasculatureSegmentId + i >= 0)
            {
                const int64_t segmentId =
                    endFoot.second.vasculatureSegmentId + i;
                if (segmentId < 0)
                    break;
                auto it = vasculatureNodes.begin();
                std::advance(it, segmentId);
                length = glm::length(centerPosition - it->second.position);
                startIndex = segmentId;
                --i;
            }

            length = 0.0;
            i = 1;
            // Now find the end segment
            while (length < halfLength &&
                   endFoot.second.vasculatureSegmentId + i <
                       vasculatureNodes.size())
            {
                const int64_t segmentId =
                    endFoot.second.vasculatureSegmentId + i;
                auto it = vasculatureNodes.begin();
                std::advance(it, segmentId);
                length = glm::length(centerPosition - it->second.position);
                endIndex = segmentId;
                ++i;
            }

            // Build the segment using spheres and cones
            uint64_t geometryIndex = 0;
            Neighbours neighbours;
            for (uint64_t i = startIndex; i < endIndex - 1; ++i)
            {
                auto it = vasculatureNodes.begin();
                std::advance(it, i);
                const auto& srcNode = it->second;
                const auto srcUserData = it->first;
                const auto srcVasculatureRadius =
                    (srcUserData < radii.size() ? radii[srcUserData]
                                                : srcNode.radius) *
                    radiusMultiplier;
                const float srcEndFootRadius =
                    DEFAULT_ENDFOOT_RADIUS_RATIO * srcVasculatureRadius;

                ++it;
                const auto& dstNode = it->second;
                const auto dstUserData = it->first;
                const auto dstVasculatureRadius =
                    (dstUserData < radii.size() ? radii[dstUserData]
                                                : dstNode.radius) *
                    radiusMultiplier;
                const float dstEndFootRadius =
                    DEFAULT_ENDFOOT_RADIUS_RATIO * dstVasculatureRadius;

                const auto srcPosition = _animatedPosition(
                    Vector4d(srcNode.position, srcVasculatureRadius));
                const auto dstPosition = _animatedPosition(
                    Vector4d(dstNode.position, dstVasculatureRadius));

                if (!_details.useSdf)
                    container.addSphere(srcPosition, srcEndFootRadius,
                                        materialId, srcUserData);
                geometryIndex =
                    container.addCone(srcPosition, srcEndFootRadius,
                                      dstPosition, dstEndFootRadius, materialId,
                                      srcUserData, neighbours, displacement);
                neighbours = {geometryIndex};
            }
        }
    }
}

void Astrocytes::setVasculatureRadiusReport(
    const VasculatureRadiusReportDetails& details)
{
    auto& connector = DBConnector::getInstance();
    const auto simulationReport =
        connector.getSimulationReport(details.populationName,
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

} // namespace morphology
} // namespace bioexplorer
