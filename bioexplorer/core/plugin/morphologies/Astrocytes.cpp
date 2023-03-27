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

#include "Astrocytes.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>
#include <plugin/common/ThreadSafeContainer.h>
#include <plugin/common/Utils.h>

#include <plugin/io/db/DBConnector.h>

#include <plugin/meshing/PointCloudMesher.h>

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
using namespace meshing;

const double DEFAULT_MITOCHONDRIA_DENSITY = 0.0459;
const double DEFAULT_ENDFOOT_RADIUS_RATIO = 1.1;
const double DEFAULT_ENDFOOT_RADIUS_SHIFTING_RATIO = 0.35;

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

double Astrocytes::_getDisplacementValue(const DisplacementElement& element)
{
    const auto params = _details.displacementParams;
    switch (element)
    {
    case DisplacementElement::morphology_soma_strength:
        return valueFromDoubles(params, 0, DEFAULT_MORPHOLOGY_SOMA_STRENGTH);
    case DisplacementElement::morphology_soma_frequency:
        return valueFromDoubles(params, 1, DEFAULT_MORPHOLOGY_SOMA_FREQUENCY);
    case DisplacementElement::morphology_section_strength:
        return valueFromDoubles(params, 2, DEFAULT_MORPHOLOGY_SECTION_STRENGTH);
    case DisplacementElement::morphology_section_frequency:
        return valueFromDoubles(params, 3,
                                DEFAULT_MORPHOLOGY_SECTION_FREQUENCY);
    case DisplacementElement::morphology_nucleus_strength:
        return valueFromDoubles(params, 4, DEFAULT_MORPHOLOGY_NUCLEUS_STRENGTH);
    case DisplacementElement::morphology_nucleus_frequency:
        return valueFromDoubles(params, 5,
                                DEFAULT_MORPHOLOGY_NUCLEUS_FREQUENCY);
    case DisplacementElement::morphology_mitochondrion_strength:
        return valueFromDoubles(params, 6,
                                DEFAULT_MORPHOLOGY_MITOCHONDRION_STRENGTH);
    case DisplacementElement::morphology_mitochondrion_frequency:
        return valueFromDoubles(params, 7,
                                DEFAULT_MORPHOLOGY_MITOCHONDRION_FREQUENCY);
    case DisplacementElement::vasculature_segment_strength:
        return valueFromDoubles(params, 8,
                                DEFAULT_VASCULATURE_SEGMENT_STRENGTH);
    case DisplacementElement::vasculature_segment_frequency:
        return valueFromDoubles(params, 9,
                                DEFAULT_VASCULATURE_SEGMENT_FREQUENCY);
    default:
        PLUGIN_THROW("Invalid displacement element");
    }
}

void Astrocytes::_logRealismParams()
{
    PLUGIN_INFO(1, "----------------------------------------------------");
    PLUGIN_INFO(1, "Realism level ("
                       << static_cast<uint32_t>(_details.realismLevel) << ")");
    PLUGIN_INFO(1, "- Soma     : " << boolAsString(andCheck(
                       static_cast<uint32_t>(_details.realismLevel),
                       static_cast<uint32_t>(MorphologyRealismLevel::soma))));
    PLUGIN_INFO(1, "- Dendrite : " << boolAsString(
                       andCheck(static_cast<uint32_t>(_details.realismLevel),
                                static_cast<uint32_t>(
                                    MorphologyRealismLevel::dendrite))));
    PLUGIN_INFO(1, "- Internals: " << boolAsString(
                       andCheck(static_cast<uint32_t>(_details.realismLevel),
                                static_cast<uint32_t>(
                                    MorphologyRealismLevel::internals))));
    PLUGIN_INFO(1, "----------------------------------------------------");
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
    const auto realismLevel = _details.realismLevel;
    const auto somas =
        connector.getAstrocytes(_details.populationName, _details.sqlFilter);
    const auto loadEndFeet = !_details.vasculaturePopulationName.empty();
    const auto loadMicroDomain = _details.loadMicroDomain;

    // Micro domain mesh per thread
    std::map<size_t, std::map<size_t, TriangleMesh>> microDomainMeshes;

    PLUGIN_INFO(1, "Building " << somas.size() << " astrocytes");
    _logRealismParams();

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

        ThreadSafeContainer container(*model, _scale);

        // Load data from DB
        double somaRadius = 0.0;
        SectionMap sections;
        if (_details.loadSomas || _details.loadDendrites)
            sections =
                connector.getAstrocyteSections(_details.populationName, somaId,
                                               !_details.loadDendrites);

        // End feet
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
            const bool useSdf =
                andCheck(static_cast<uint32_t>(_details.realismLevel),
                         static_cast<uint32_t>(MorphologyRealismLevel::soma));
            somaGeometryIndex = container.addSphere(
                somaPosition, somaRadius, somaMaterialId, useSdf, NO_USER_DATA,
                {},
                Vector3f(
                    somaRadius *
                        _getDisplacementValue(
                            DisplacementElement::morphology_soma_strength),
                    somaRadius *
                        _getDisplacementValue(
                            DisplacementElement::morphology_soma_frequency),
                    0.f));
            if (_details.generateInternals)
            {
                const auto useSdf =
                    andCheck(static_cast<uint32_t>(_details.realismLevel),
                             static_cast<uint32_t>(
                                 MorphologyRealismLevel::internals));
                _addSomaInternals(container, baseMaterialId, somaPosition,
                                  somaRadius, DEFAULT_MITOCHONDRIA_DENSITY,
                                  useSdf);
            }
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
                const bool useSdf =
                    andCheck(static_cast<uint32_t>(_details.realismLevel),
                             static_cast<uint32_t>(
                                 MorphologyRealismLevel::dendrite));
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
                        somaMaterialId, useSdf, userData, neighbours,
                        Vector3f(srcRadius * _getDisplacementValue(
                                                 DisplacementElement::
                                                     morphology_soma_strength),
                                 srcRadius * _getDisplacementValue(
                                                 DisplacementElement::
                                                     morphology_soma_frequency),
                                 0.f));
                    neighbours.insert(geometryIndex);
                }

                // If maxDistanceToSoma != 0, then compute actual distance from
                // soma
                double distanceToSoma = 0.0;
                if (_details.maxDistanceToSoma > 0.0)
                    distanceToSoma =
                        _getDistanceToSoma(sections, section.second);
                if (distanceToSoma > _details.maxDistanceToSoma)
                    continue;

                // Process section points according to representation
                const auto localPoints = _getProcessedSectionPoints(
                    _details.morphologyRepresentation, points);

                double sectionLength = 0.0;
                for (uint64_t i = 0; i < localPoints.size() - 1; i += step)
                {
                    const auto srcPoint = localPoints[i];
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
                        dstPoint = localPoints[i + step];
                        dstRadius = dstPoint.w * 0.5 * _radiusMultiplier;
                        ++i;
                    } while (length(Vector3f(dstPoint) - Vector3f(srcPoint)) <
                                 (srcRadius + dstRadius) &&
                             (i + step) < localPoints.size() - 1);
                    --i;

                    // Distance to soma
                    sectionLength += length(dstPoint - srcPoint);
                    _maxDistanceToSoma =
                        std::max(_maxDistanceToSoma,
                                 distanceToSoma + sectionLength);

                    const size_t materialId =
                        _details.morphologyColorScheme ==
                                MorphologyColorScheme::distance_to_soma
                            ? _getMaterialFromDistanceToSoma(
                                  _details.maxDistanceToSoma, distanceToSoma)

                            : sectionMaterialId;

                    const auto dst = _animatedPosition(
                        Vector4d(somaPosition + Vector3d(dstPoint), dstRadius),
                        somaId);
                    if (!useSdf)
                        geometryIndex =
                            container.addSphere(dst, dstRadius, materialId,
                                                useSdf, NO_USER_DATA);

                    geometryIndex = container.addCone(
                        src, srcRadius, dst, dstRadius, materialId, useSdf,
                        userData, {geometryIndex},
                        Vector3f(srcRadius *
                                     _getDisplacementValue(
                                         DisplacementElement::
                                             morphology_section_strength),
                                 _getDisplacementValue(
                                     DisplacementElement::
                                         morphology_section_frequency),
                                 0.f));

                    _bounds.merge(srcPoint);

                    if (_details.maxDistanceToSoma > 0.0 &&
                        distanceToSoma + sectionLength >=
                            _details.maxDistanceToSoma)
                        break;
                }
            }
        }

        if (loadEndFeet)
            _addEndFoot(container, soma.center, endFeet, radii, baseMaterialId);

        if (loadMicroDomain)
        {
            const auto materialId =
                (_details.morphologyColorScheme ==
                 MorphologyColorScheme::section_type)
                    ? baseMaterialId + MATERIAL_OFFSET_MICRO_DOMAIN
                    : baseMaterialId;

            switch (_details.microDomainRepresentation)
            {
            case MicroDomainRepresentation::convex_hull:
            {
                _buildMicroDomain(container, somaId, materialId);
                break;
            }
            default:
            {
                auto& mesh = microDomainMeshes[index][materialId];
                _addMicroDomain(mesh, somaId);
                break;
            }
            }
        }
#pragma omp critical
        containers.push_back(container);
    }

    for (uint64_t i = 0; i < containers.size(); ++i)
    {
        PLUGIN_PROGRESS("- Compiling 3D geometry...", i, containers.size());
        auto& container = containers[i];
        if (_details.microDomainRepresentation ==
            MicroDomainRepresentation::mesh)
            for (const auto& mesh : microDomainMeshes[i])
                container.addMesh(mesh.first, mesh.second);
        container.commitToModel();
    }

    const ModelMetadata metadata = {{"Number of astrocytes",
                                     std::to_string(somas.size())},
                                    {"SQL filter", _details.sqlFilter},
                                    {"Max distance to soma",
                                     std::to_string(_maxDistanceToSoma)}};

    _modelDescriptor.reset(new brayns::ModelDescriptor(std::move(model),
                                                       _details.assemblyName,
                                                       metadata));
    if (_modelDescriptor)
        _scene.addModel(_modelDescriptor);
    else
        PLUGIN_THROW("Astrocytes model could not be created");
}

void Astrocytes::_addEndFoot(ThreadSafeContainer& container,
                             const Vector3d& somaCenter,
                             const EndFootMap& endFeet, const doubles& radii,
                             const size_t baseMaterialId)
{
    const auto radiusMultiplier = _details.radiusMultiplier;
    const Vector3d displacement{
        _getDisplacementValue(
            DisplacementElement::vasculature_segment_strength),
        _getDisplacementValue(
            DisplacementElement::vasculature_segment_frequency),
        0.0};
    const auto useSdf =
        andCheck(static_cast<uint32_t>(_details.realismLevel),
                 static_cast<uint32_t>(MorphologyRealismLevel::end_foot));

    const auto materialId =
        (_details.morphologyColorScheme == MorphologyColorScheme::section_type)
            ? baseMaterialId + MATERIAL_OFFSET_END_FOOT
            : baseMaterialId;

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
            // Find start segment making the assumption that the segment Id
            // is in the middle of the end-foot
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
            uint64_t endFootSegmentIndex = 0;
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

                // Shift position in direction of astrocyte soma, so that
                // only half of the end-feet appears outside of the
                // vasculature
                const Vector3d& shift =
                    normalize(srcNode.position - somaCenter) *
                    srcVasculatureRadius *
                    DEFAULT_ENDFOOT_RADIUS_SHIFTING_RATIO *
                    (1.0 + rnd2(endFootSegmentIndex));

                const auto srcPosition = _animatedPosition(
                    Vector4d(srcNode.position - shift, srcVasculatureRadius));
                const auto dstPosition = _animatedPosition(
                    Vector4d(dstNode.position - shift, dstVasculatureRadius));

                if (!useSdf)
                    container.addSphere(srcPosition, srcEndFootRadius,
                                        materialId, useSdf, srcUserData);
                geometryIndex =
                    container.addCone(srcPosition, srcEndFootRadius,
                                      dstPosition, dstEndFootRadius, materialId,
                                      useSdf, srcUserData, neighbours,
                                      displacement);
                neighbours = {geometryIndex};
                ++endFootSegmentIndex;
            }
        }
    }
}

void Astrocytes::_addMicroDomain(TriangleMesh& dstMesh,
                                 const uint64_t astrocyteId)
{
    auto& connector = DBConnector::getInstance();
    const auto srcMesh =
        connector.getAstrocyteMicroDomain(_details.populationName, astrocyteId);
    auto vertexOffset = dstMesh.vertices.size();
    dstMesh.vertices.insert(dstMesh.vertices.end(), srcMesh.vertices.begin(),
                            srcMesh.vertices.end());

    auto indexOffset = dstMesh.indices.size();
    dstMesh.indices.insert(dstMesh.indices.end(), srcMesh.indices.begin(),
                           srcMesh.indices.end());
    for (uint64_t i = 0; i < srcMesh.indices.size(); ++i)
        dstMesh.indices[indexOffset + i] += vertexOffset;
    dstMesh.normals.insert(dstMesh.normals.end(), srcMesh.normals.begin(),
                           srcMesh.normals.end());
    dstMesh.colors.insert(dstMesh.colors.end(), srcMesh.colors.begin(),
                          srcMesh.colors.end());
}

void Astrocytes::_buildMicroDomain(ThreadSafeContainer& container,
                                   const uint64_t astrocyteId,
                                   const size_t materialId)
{
    auto& connector = DBConnector::getInstance();
    const auto mesh =
        connector.getAstrocyteMicroDomain(_details.populationName, astrocyteId);

    PointCloud cloud;
    for (const auto& v : mesh.vertices)
        cloud[materialId].push_back(Vector4d(v.x, v.y, v.z, 0.5));

    PointCloudMesher pcm;
    if (!pcm.toConvexHull(container, cloud))
        PLUGIN_THROW(
            "Failed to generate convex hull from micro domain information");
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
