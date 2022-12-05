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

#include "Neurons.h"
#include "CompartmentSimulationHandler.h"
#include "SomaSimulationHandler.h"
#include "SpikeSimulationHandler.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>
#include <plugin/common/shapes/Shape.h>

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

const uint64_t NB_MYELIN_FREE_SEGMENTS = 4;
const double DEFAULT_SPINE_RADIUS = 0.25;
const double DEFAULT_ARROW_RADIUS_RATIO = 10.0;
const Vector2d DEFAULT_SIMULATION_VALUE_RANGE = {-80.0, -10.0};

// Mitochondria density per layer
// Source: A simplified morphological classification scheme for pyramidal cells
// in six layers of primary somatosensory cortex of juvenile rats
// https://www.sciencedirect.com/science/article/pii/S2451830118300293)
const doubles MITOCHONDRIA_DENSITY = {0.0459, 0.0522, 0.064,
                                      0.0774, 0.0575, 0.0403};

Neurons::Neurons(Scene& scene, const NeuronsDetails& details)
    : Morphologies(details.radiusMultiplier, doublesToVector3d(details.scale))
    , _details(details)
    , _scene(scene)
{
    _radiusMultiplier =
        _details.radiusMultiplier > 0.0 ? _details.radiusMultiplier : 1.0;
    _animationDetails = doublesToCellAnimationDetails(_details.animationParams);
    srand(_animationDetails.seed);

    Timer chrono;
    _buildNeurons();
    PLUGIN_TIMER(chrono.elapsed(), "Neurons loaded");
}

void Neurons::_buildNeurons()
{
    const auto& connector = DBConnector::getInstance();

    auto model = _scene.createModel();
    const auto useSdf = ((_details.morphologyRepresentation ==
                              MorphologyRepresentation::graph ||
                          _details.morphologyRepresentation ==
                              MorphologyRepresentation::orientation)
                             ? false
                             : _details.useSdf);

    // Simulation report
    const auto sqlNodeFilter = _attachSimulationReport(*model);

    // Neurons
    const auto somas =
        connector.getNeurons(_details.populationName, sqlNodeFilter);

    PLUGIN_INFO(1, "Building " << somas.size() << " neurons");

    size_t previousMaterialId = std::numeric_limits<size_t>::max();
    size_t baseMaterialId = 0;
    Vector3ui indexOffset;

    const bool somasOnly = _details.loadSomas && !_details.loadAxon &&
                           !_details.loadApicalDendrites &&
                           !_details.loadBasalDendrites;

    ThreadSafeContainers containers;
    if (somasOnly || _details.morphologyRepresentation ==
                         MorphologyRepresentation::orientation)
    {
        ThreadSafeContainer container(*model, useSdf, _scale);
        if (_details.morphologyRepresentation ==
            MorphologyRepresentation::orientation)
            _buildOrientations(container, somas, baseMaterialId);
        else
            _buildSomasOnly(container, somas, baseMaterialId);
        containers.push_back(container);
    }
    else
    {
        const auto nbDBConnections =
            DBConnector::getInstance().getNbConnections();

        uint64_t neuronIndex;
#pragma omp parallel for num_threads(nbDBConnections)
        for (neuronIndex = 0; neuronIndex < somas.size(); ++neuronIndex)
        {
            if (omp_get_thread_num() == 0)
                PLUGIN_PROGRESS("Loading neurons", neuronIndex,
                                somas.size() / nbDBConnections);

            auto it = somas.begin();
            std::advance(it, neuronIndex);
            const auto& soma = it->second;
            ThreadSafeContainer container(*model, useSdf, _scale);
            _buildMorphology(container, it->first, soma, neuronIndex);

#pragma omp critical
            containers.push_back(container);
        }
    }

    for (uint64_t i = 0; i < containers.size(); ++i)
    {
        PLUGIN_PROGRESS("- Compiling 3D geometry...", i, containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }

    ModelMetadata metadata = {
        {"Number of Neurons", std::to_string(somas.size())},
        {"Number of Spines", std::to_string(_nbSpines)},
    };

    _modelDescriptor.reset(new brayns::ModelDescriptor(std::move(model),
                                                       _details.assemblyName,
                                                       metadata));
    if (_modelDescriptor)
    {
        _scene.addModel(_modelDescriptor);
        PLUGIN_INFO(1, "Successfully loaded " << somas.size() << " neurons");
    }
    else
        PLUGIN_THROW("Neurons model could not be created");
}

void Neurons::_buildSomasOnly(ThreadSafeContainer& container,
                              const NeuronSomaMap& somas,
                              const size_t baseMaterialId)
{
    uint64_t progress = 0;
    uint64_t neuronIndex = 0;
    for (const auto soma : somas)
    {
        PLUGIN_PROGRESS("Loading somas", progress, somas.size());
        const auto somaMaterialId =
            baseMaterialId + (_details.morphologyColorScheme ==
                                      MorphologyColorScheme::section_type
                                  ? MATERIAL_OFFSET_SOMA
                                  : 0);
        if (_details.showMembrane)
            container.addSphere(soma.second.position, _details.radiusMultiplier,
                                somaMaterialId, neuronIndex, {},
                                Vector3f(neuronSomaDisplacementStrength,
                                         neuronSomaDisplacementFrequency, 0.f));
        if (_details.generateInternals)
        {
            const double mitochondriaDensity =
                (soma.second.layer < MITOCHONDRIA_DENSITY.size()
                     ? MITOCHONDRIA_DENSITY[soma.second.layer]
                     : 0.0);

            _addSomaInternals(container, baseMaterialId, soma.second.position,
                              _details.radiusMultiplier, mitochondriaDensity);
        }
        ++progress;
        ++neuronIndex;
    }
}

void Neurons::_buildOrientations(ThreadSafeContainer& container,
                                 const NeuronSomaMap& somas,
                                 const size_t baseMaterialId)
{
    const auto radius = _details.radiusMultiplier;
    uint64_t progress = 0;
    for (const auto soma : somas)
    {
        PLUGIN_PROGRESS("Loading soma orientations", progress, somas.size());
        _addArrow(container, soma.first, soma.second.position,
                  soma.second.rotation, Vector4d(0, 0, 0, radius * 0.2),
                  Vector4d(radius, 0, 0, radius * 0.2), NeuronSectionType::soma,
                  0);
        ++progress;
    }
}

void Neurons::_buildMorphology(ThreadSafeContainer& container,
                               const uint64_t neuronId, const NeuronSoma& soma,
                               const uint64_t neuronIndex)
{
    const auto& connector = DBConnector::getInstance();

    const auto& somaRotation = soma.rotation;
    const auto layer = soma.layer;
    const double mitochondriaDensity =
        (layer < MITOCHONDRIA_DENSITY.size() ? MITOCHONDRIA_DENSITY[layer]
                                             : 0.0);

    SectionMap sections;
    // Soma radius
    double somaRadius = _radiusMultiplier;
    if (_details.loadAxon || _details.loadApicalDendrites ||
        _details.loadBasalDendrites)
    {
        sections =
            connector.getNeuronSections(_details.populationName, neuronId,
                                        _details.sqlSectionFilter);
        uint64_t count = 0;
        for (const auto& section : sections)
            if (section.second.parentId == SOMA_AS_PARENT)
            {
                const auto point = section.second.points[0];
                somaRadius += 0.5 * length(point);
                ++count;
            }
        if (count > 0)
            somaRadius = _radiusMultiplier * somaRadius / count;
    }
    const auto somaPosition =
        _animatedPosition(Vector4d(soma.position, somaRadius), neuronId);

    size_t baseMaterialId;
    switch (_details.populationColorScheme)
    {
    case PopulationColorScheme::none:
        baseMaterialId = 0;
        break;
    case PopulationColorScheme::id:
        baseMaterialId = neuronIndex * NB_MATERIALS_PER_MORPHOLOGY;
        break;
    }

    size_t somaMaterialId;
    switch (_details.morphologyColorScheme)
    {
    case MorphologyColorScheme::none:
        somaMaterialId = baseMaterialId;
        break;
    case MorphologyColorScheme::section_type:
        somaMaterialId = baseMaterialId + MATERIAL_OFFSET_SOMA;
        break;
    case MorphologyColorScheme::section_orientation:
        somaMaterialId = getMaterialIdFromOrientation({1.0, 1.0, 1.0});
        break;
    }

    // Soma
    uint64_t somaUserData = NO_USER_DATA;
    switch (_reportType)
    {
    case ReportType::compartment:
    {
        const auto compartments =
            connector.getNeuronSectionCompartments(_details.populationName,
                                                   _details.simulationReportId,
                                                   neuronId, 0);
        if (!compartments.empty())
            somaUserData = compartments[0];
        break;
    }
    case ReportType::soma:
    {
        somaUserData = neuronIndex;
        break;
    }
    }

    uint64_t somaGeometryIndex = 0;
    if (_details.loadSomas)
    {
        if (_details.showMembrane)
            somaGeometryIndex =
                container.addSphere(somaPosition, somaRadius, somaMaterialId,
                                    somaUserData, {},
                                    Vector3f(neuronSomaDisplacementStrength,
                                             neuronSomaDisplacementFrequency,
                                             0.f));
        if (_details.generateInternals)
            _addSomaInternals(container, baseMaterialId, somaPosition,
                              somaRadius, mitochondriaDensity);
    }

    // Load synapses for all sections
    SectionSynapseMap synapses;
    if (_details.loadSynapses)
        synapses =
            connector.getNeuronSynapses(_details.populationName, neuronId);

    // Sections (dendrites and axon)
    uint64_t geometryIndex = 0;
    Neighbours neighbours{somaGeometryIndex};

    for (const auto& section : sections)
    {
        const auto sectionType =
            static_cast<NeuronSectionType>(section.second.type);
        const auto& points = section.second.points;
        if (sectionType == NeuronSectionType::axon && !_details.loadAxon)
            continue;
        if (sectionType == NeuronSectionType::basal_dendrite &&
            !_details.loadBasalDendrites)
            continue;
        if (sectionType == NeuronSectionType::apical_dendrite &&
            !_details.loadApicalDendrites)
            continue;

        if (_details.morphologyRepresentation ==
            MorphologyRepresentation::graph)
        {
            _addArrow(container, neuronIndex, somaPosition, somaRotation,
                      section.second.points[0],
                      section.second.points[section.second.points.size() - 1],
                      sectionType, baseMaterialId);
            continue;
        }

        // Sections connected to the soma
        if (_details.showMembrane && _details.loadSomas &&
            section.second.parentId == SOMA_AS_PARENT)
        {
            const auto& point = section.second.points[0];

            const float srcRadius = somaRadius * 0.75f * _radiusMultiplier;
            const float dstRadius = point.w * 0.5f * _radiusMultiplier;

            const auto sectionType =
                static_cast<NeuronSectionType>(section.second.type);
            const bool loadSection =
                (sectionType == NeuronSectionType::axon && _details.loadAxon) ||
                (sectionType == NeuronSectionType::basal_dendrite &&
                 _details.loadBasalDendrites) ||
                (sectionType == NeuronSectionType::apical_dendrite &&
                 _details.loadApicalDendrites);

            if (!loadSection)
                continue;

            const auto dstPosition =
                _animatedPosition(Vector4d(somaPosition +
                                               somaRotation * Vector3d(point),
                                           dstRadius),
                                  neuronId);
            const Vector3f displacement{dstRadius * sectionDisplacementStrength,
                                        sectionDisplacementFrequency, 0.f};
            geometryIndex =
                container.addCone(somaPosition, srcRadius, dstPosition,
                                  dstRadius, somaMaterialId, somaUserData,
                                  neighbours, displacement);
            neighbours.insert(geometryIndex);
        }

        _addSection(container, neuronId, soma.morphologyId, section.first,
                    section.second, geometryIndex, somaPosition, somaRotation,
                    somaRadius, baseMaterialId, mitochondriaDensity,
                    somaUserData, synapses);
    }
}

void Neurons::_addVaricosity(Vector4fs& points)
{
    // Reference: The cholinergic innervation develops early and rapidly in the
    // rat cerebral cortex: a quantitative immunocytochemical study
    // https://www.sciencedirect.com/science/article/abs/pii/S030645220100389X
    const uint64_t middlePointIndex = points.size() / 2;
    const auto& startPoint = points[middlePointIndex];
    const auto& endPoint = points[middlePointIndex + 1];
    const double radius = std::min(startPoint.w, startPoint.w);

    const auto sp = Vector3d(startPoint);
    const auto ep = Vector3d(endPoint);

    const Vector3d dir = ep - sp;
    const Vector3d p0 = sp + dir * 0.2;
    const Vector3d p1 = sp + dir * 0.5 +
                        radius * Vector3d((rand() % 100 - 50) / 100.0,
                                          (rand() % 100 - 50) / 100.0,
                                          (rand() % 100 - 50) / 100.0);
    const Vector3d p2 = sp + dir * 0.8;

    auto idx = points.begin() + middlePointIndex + 1;
    idx = points.insert(idx, {p2.x, p2.y, p2.z, startPoint.w});
    idx = points.insert(idx, {p1.x, p1.y, p1.z, radius * 2.0});
    points.insert(idx, {p0.x, p0.y, p0.z, endPoint.w});
}

void Neurons::_addArrow(ThreadSafeContainer& container, const uint64_t neuronId,
                        const Vector3d& somaPosition,
                        const Quaterniond& somaRotation,
                        const Vector4d& srcNode, const Vector4d& dstNode,
                        const NeuronSectionType sectionType,
                        const size_t baseMaterialId)
{
    size_t sectionMaterialId;
    switch (_details.morphologyColorScheme)
    {
    case MorphologyColorScheme::none:
        sectionMaterialId = sectionMaterialId = baseMaterialId;
        break;
    case MorphologyColorScheme::section_type:
        sectionMaterialId = baseMaterialId + static_cast<size_t>(sectionType);
        break;
    case MorphologyColorScheme::section_orientation:
        sectionMaterialId =
            getMaterialIdFromOrientation(somaRotation * Vector3d(0, 0, 1));
        break;
    }

    const auto src = _animatedPosition(
        Vector4d(somaPosition + somaRotation * Vector3d(srcNode), srcNode.w),
        neuronId);
    const auto dst = _animatedPosition(
        Vector4d(somaPosition + somaRotation * Vector3d(dstNode), dstNode.w),
        neuronId);
    const auto userData = neuronId;
    const auto direction = dst - src;
    const auto maxRadius = std::max(srcNode.w, dstNode.w);
    const float radius = std::min(length(direction) / 5.0,
                                  maxRadius * _details.radiusMultiplier);
    container.addSphere(src, radius * 0.2, sectionMaterialId, userData);
    container.addCone(src, radius * 0.2, Vector3f(src + direction * 0.79),
                      radius * 0.2, sectionMaterialId, userData);
    container.addCone(Vector3f(src + direction * 0.79), radius * 0.2,
                      Vector3f(src + direction * 0.8), radius,
                      sectionMaterialId, userData);
    container.addCone(Vector3f(src + direction * 0.8), radius, dst, 0.0,
                      sectionMaterialId, userData);
    _bounds.merge(src);
    _bounds.merge(dst);
}

void Neurons::_addSection(ThreadSafeContainer& container,
                          const uint64_t neuronId, const uint64_t morphologyId,
                          const uint64_t sectionId, const Section& section,
                          const uint64_t somaGeometryIndex,
                          const Vector3d& somaPosition,
                          const Quaterniond& somaRotation,
                          const double somaRadius, const size_t baseMaterialId,
                          const double mitochondriaDensity,
                          const uint64_t somaUserData,
                          const SectionSynapseMap& synapses)
{
    const auto& connector = DBConnector::getInstance();
    const auto sectionType = static_cast<NeuronSectionType>(section.type);
    auto useSdf = _details.useSdf;
    auto userData = NO_USER_DATA;

    const auto& points = section.points;
    size_t sectionMaterialId;
    switch (_details.morphologyColorScheme)
    {
    case MorphologyColorScheme::none:
        sectionMaterialId = baseMaterialId;
        break;
    case MorphologyColorScheme::section_type:
        sectionMaterialId = baseMaterialId + section.type;
        break;
    case MorphologyColorScheme::section_orientation:
        sectionMaterialId =
            getMaterialIdFromOrientation(points[points.size() - 1] - points[0]);
        break;
    }

    // Generate varicosities
    auto localPoints = points;
    const auto middlePointIndex = localPoints.size() / 2;
    const bool addVaricosity = _details.generateVaricosities &&
                               sectionType == NeuronSectionType::axon &&
                               localPoints.size() > nbMinSegmentsForVaricosity;

    if (addVaricosity)
        _addVaricosity(localPoints);

    // Section surface
    double sectionLength = 0.0;
    double sectionVolume = 0.0;
    uint64_t geometryIndex = 0;
    Neighbours neighbours;
    if (_details.morphologyColorScheme == MorphologyColorScheme::none)
        neighbours.insert(somaGeometryIndex);

    uint64_ts compartments;
    switch (_reportType)
    {
    case ReportType::undefined:
        userData = NO_USER_DATA;
        break;
    case ReportType::spike:
    case ReportType::soma:
    {
        userData = somaUserData;
        break;
    }
    case ReportType::compartment:
    {
        compartments =
            connector.getNeuronSectionCompartments(_details.populationName,
                                                   _details.simulationReportId,
                                                   neuronId, sectionId);
        break;
    }
    }

    // Section synapses
    SegmentSynapseMap segmentSynapses;
    const auto it = synapses.find(sectionId);
    if (it != synapses.end())
        segmentSynapses = (*it).second;

    for (uint64_t i = 0; i < localPoints.size() - 1; ++i)
    {
        if (!compartments.empty())
        {
            const uint64_t compartmentIndex =
                i * compartments.size() / localPoints.size();
            userData = compartments[compartmentIndex];
        }

        const auto& srcPoint = localPoints[i];
        const float srcRadius = srcPoint.w * 0.5f * _radiusMultiplier;
        const auto src =
            _animatedPosition(Vector4d(somaPosition +
                                           somaRotation * Vector3d(srcPoint),
                                       srcRadius),
                              neuronId);

        const auto& dstPoint = localPoints[i + 1];
        const float dstRadius = dstPoint.w * 0.5f * _radiusMultiplier;
        const auto dst =
            _animatedPosition(Vector4d(somaPosition +
                                           somaRotation * Vector3d(dstPoint),
                                       dstRadius),
                              neuronId);
        const double sampleLength = length(dstPoint - srcPoint);
        sectionLength += sampleLength;

        if (_details.showMembrane)
        {
            if (i > 0 && _details.morphologyRepresentation !=
                             MorphologyRepresentation::segment)
                neighbours = {geometryIndex};

            Vector3f displacement{srcRadius * sectionDisplacementStrength,
                                  sectionDisplacementFrequency, 0.f};
            size_t materialId = sectionMaterialId;
            if (addVaricosity && _details.morphologyColorScheme ==
                                     MorphologyColorScheme::section_type)
            {
                if (i > middlePointIndex && i < middlePointIndex + 3)
                {
                    materialId = baseMaterialId + MATERIAL_OFFSET_VARICOSITY;
                    displacement =
                        Vector3f(2.f * srcRadius * sectionDisplacementStrength,
                                 sectionDisplacementFrequency, 0.f);
                }
                if (i == middlePointIndex + 1 || i == middlePointIndex + 3)
                    neighbours = {};
                if (i == middlePointIndex + 1)
                    _varicosities[neuronId].push_back(dst);
            }

            if (!useSdf)
                container.addSphere(dst, dstRadius, materialId, userData);

            const auto it = segmentSynapses.find(i);
            if (it != segmentSynapses.end())
            {
                const size_t spineMaterialId =
                    _details.morphologyColorScheme ==
                            MorphologyColorScheme::section_type
                        ? baseMaterialId + MATERIAL_OFFSET_SYNPASE
                        : materialId;
                const auto synapses = (*it).second;
                PLUGIN_DEBUG("Adding " << synapses.size()
                                       << " spines to segment " << i
                                       << " of section " << sectionId);
                for (const auto& synapse : synapses)
                {
                    const Vector3d segmentDirection = normalize(dst - src);
                    const Vector3d surfacePosition =
                        src +
                        segmentDirection * synapse.preSynapticSegmentDistance;
                    _addSpine(container, neuronId, morphologyId, sectionId,
                              synapse, spineMaterialId, surfacePosition);
                }
            }

            geometryIndex =
                container.addCone(src, srcRadius, dst, dstRadius, materialId,
                                  userData, neighbours, displacement);

            neighbours.insert(geometryIndex);
        }
        sectionVolume += coneVolume(sampleLength, srcRadius, dstRadius);

        _bounds.merge(srcPoint);
    }

    if (sectionType == NeuronSectionType::axon)
    {
        if (_details.generateInternals)
            _addSectionInternals(container, neuronId, somaPosition,
                                 somaRotation, sectionLength, sectionVolume,
                                 points, mitochondriaDensity, baseMaterialId);

        if (_details.generateExternals)
            _addAxonMyelinSheath(container, neuronId, somaPosition,
                                 somaRotation, sectionLength, points,
                                 mitochondriaDensity, baseMaterialId);
    }
}

void Neurons::_addSectionInternals(
    ThreadSafeContainer& container, const uint64_t neuronId,
    const Vector3d& somaPosition, const Quaterniond& somaRotation,
    const double sectionLength, const double sectionVolume,
    const Vector4fs& points, const double mitochondriaDensity,
    const size_t baseMaterialId)
{
    if (mitochondriaDensity == 0.0)
        return;

    const auto useSdf = _details.useSdf;

    // Add mitochondria (density is per section, not for the full axon)
    const size_t nbMaxMitochondrionSegments =
        sectionLength / mitochondrionSegmentSize;
    const double indexRatio =
        double(points.size()) / double(nbMaxMitochondrionSegments);

    double mitochondriaVolume = 0.0;
    const size_t mitochondrionMaterialId =
        baseMaterialId + MATERIAL_OFFSET_MITOCHONDRION;

    uint64_t nbSegments = _getNbMitochondrionSegments();
    int64_t mitochondrionSegment =
        -(rand() % (1 + nbMaxMitochondrionSegments / 10));
    double previousRadius;
    Vector3d previousPosition;

    uint64_t geometryIndex = 0;
    Vector3d randomPosition{points[0].w * (rand() % 100 - 50) / 200.0,
                            points[0].w * (rand() % 100 - 50) / 200.0,
                            points[0].w * (rand() % 100 - 50) / 200.0};
    for (size_t step = 0; step < nbMaxMitochondrionSegments; ++step)
    {
        if (mitochondriaVolume < sectionVolume * mitochondriaDensity &&
            mitochondrionSegment >= 0 && mitochondrionSegment < nbSegments)
        {
            const uint64_t srcIndex = uint64_t(step * indexRatio);
            const uint64_t dstIndex = uint64_t(step * indexRatio) + 1;
            if (dstIndex < points.size())
            {
                const auto srcSample =
                    _animatedPosition(points[srcIndex], neuronId);
                const auto dstSample =
                    _animatedPosition(points[dstIndex], neuronId);
                const double srcRadius =
                    points[srcIndex].w * 0.5 * _radiusMultiplier;
                const Vector3d srcPosition{
                    srcSample.x + srcRadius * (rand() % 100 - 50) / 500.0,
                    srcSample.y + srcRadius * (rand() % 100 - 50) / 500.0,
                    srcSample.z + srcRadius * (rand() % 100 - 50) / 500.0};
                const double dstRadius =
                    points[dstIndex].w * 0.5 * _radiusMultiplier;
                const Vector3d dstPosition{
                    dstSample.x + dstRadius * (rand() % 100 - 50) / 500.0,
                    dstSample.y + dstRadius * (rand() % 100 - 50) / 500.0,
                    dstSample.z + dstRadius * (rand() % 100 - 50) / 500.0};

                const Vector3d direction = dstPosition - srcPosition;
                const Vector3d position =
                    srcPosition + randomPosition +
                    direction * (step * indexRatio - srcIndex);
                const double radius =
                    (1.0 + (rand() % 1000 - 500) / 5000.0) *
                    mitochondrionRadius *
                    0.5; // Make twice smaller than in the soma

                Neighbours neighbours;
                if (mitochondrionSegment != 0)
                    neighbours = {geometryIndex};

                if (!useSdf)
                    container.addSphere(somaPosition + somaRotation * position,
                                        radius, mitochondrionMaterialId,
                                        NO_USER_DATA);

                if (mitochondrionSegment > 0)
                {
                    Neighbours neighbours = {};
                    if (mitochondrionSegment > 1)
                        neighbours = {geometryIndex};
                    const auto srcPosition =
                        _animatedPosition(Vector4d(somaPosition +
                                                       somaRotation * position,
                                                   radius),
                                          neuronId);
                    const auto dstPosition = _animatedPosition(
                        Vector4d(somaPosition + somaRotation * previousPosition,
                                 previousRadius),
                        neuronId);
                    geometryIndex = container.addCone(
                        srcPosition, radius, dstPosition, previousRadius,
                        mitochondrionMaterialId, NO_USER_DATA, neighbours,
                        Vector3f(radius * mitochondrionDisplacementStrength *
                                     2.0,
                                 radius * mitochondrionDisplacementFrequency,
                                 0.f));

                    mitochondriaVolume +=
                        coneVolume(length(position - previousPosition), radius,
                                   previousRadius);
                }

                previousPosition = position;
                previousRadius = radius;
            }
        }
        ++mitochondrionSegment;

        if (mitochondrionSegment == nbSegments)
        {
            mitochondrionSegment =
                -(rand() % (1 + nbMaxMitochondrionSegments / 10));
            nbSegments = _getNbMitochondrionSegments();
            const auto index = uint64_t(step * indexRatio);
            randomPosition =
                Vector3d(points[index].w * (rand() % 100 - 50) / 200.0,
                         points[index].w * (rand() % 100 - 50) / 200.0,
                         points[index].w * (rand() % 100 - 50) / 200.0);
        }
    }
}

void Neurons::_addAxonMyelinSheath(
    ThreadSafeContainer& container, const uint64_t neuronId,
    const Vector3d& somaPosition, const Quaterniond& somaRotation,
    const double sectionLength, const Vector4fs& points,
    const double mitochondriaDensity, const size_t baseMaterialId)
{
    if (sectionLength == 0 || points.empty())
        return;

    const auto useSdf = _details.useSdf;

    const size_t myelinSteathMaterialId =
        baseMaterialId + MATERIAL_OFFSET_MYELIN_SHEATH;

    if (sectionLength < 2 * myelinSteathLength)
        return;

    const uint64_t nbPoints = points.size();
    if (nbPoints < NB_MYELIN_FREE_SEGMENTS)
        return;

    uint64_t i = NB_MYELIN_FREE_SEGMENTS; // Ignore first 3 segments
    while (i < nbPoints - NB_MYELIN_FREE_SEGMENTS)
    {
        // Start surrounding segments with myelin steaths
        const auto& srcPoint = points[i];
        const auto srcRadius =
            srcPoint.w * 0.5 * myelinSteathRadiusRatio * _radiusMultiplier;
        const auto srcPosition =
            _animatedPosition(Vector4d(somaPosition +
                                           somaRotation * Vector3d(srcPoint),
                                       srcRadius),
                              neuronId);

        if (!useSdf)
            container.addSphere(srcPosition, srcRadius, myelinSteathMaterialId,
                                NO_USER_DATA);

        double currentLength = 0;
        auto previousPosition = srcPosition;
        auto previousRadius = srcRadius;
        const Vector3f displacement{srcRadius *
                                        myelinSteathDisplacementStrength,
                                    myelinSteathDisplacementFrequency, 0.f};
        Neighbours neighbours;

        while (currentLength < myelinSteathLength &&
               i < nbPoints - NB_MYELIN_FREE_SEGMENTS)
        {
            ++i;
            const auto& dstPoint = points[i];
            const auto dstRadius =
                dstPoint.w * 0.5 * myelinSteathRadiusRatio * _radiusMultiplier;
            const auto dstPosition = _animatedPosition(
                Vector4d(somaPosition + somaRotation * Vector3d(dstPoint),
                         dstRadius),
                neuronId);

            currentLength += length(dstPosition - previousPosition);
            if (!useSdf)
                container.addSphere(dstPosition, srcRadius,
                                    myelinSteathMaterialId, NO_USER_DATA);

            const auto geometryIndex =
                container.addCone(dstPosition, dstRadius, previousPosition,
                                  previousRadius, myelinSteathMaterialId,
                                  NO_USER_DATA, neighbours, displacement);
            neighbours.insert(geometryIndex);
            previousPosition = dstPosition;
            previousRadius = dstRadius;
        }
        i += NB_MYELIN_FREE_SEGMENTS; // Leave free segments between myelin
                                      // steaths
    }
}

void Neurons::_addSpine(ThreadSafeContainer& container, const uint64_t neuronId,
                        const uint64_t morphologyId, const uint64_t sectionId,
                        const Synapse& synapse, const size_t SpineMaterialId,
                        const Vector3d& preSynapticSurfacePosition)
{
    const double radius = DEFAULT_SPINE_RADIUS;

    // Spine geometry
#if 0
    const auto& connector = DBConnector::getInstance();
    const auto postSynapticSections = connector.getNeuronSections(
        _details.populationName, synapse.postSynapticNeuronId,
        "s.section_guid=" + std::to_string(synapse.postSynapticSectionId));

    if (postSynapticSections.empty())
    {
        PLUGIN_ERROR("Spine: " << neuronId << " / " << sectionId << " -> "
                               << synapse.postSynapticNeuronId << " / "
                               << synapse.postSynapticSectionId << " / "
                               << synapse.postSynapticSegmentId);
        PLUGIN_ERROR("Could not find section " << synapse.postSynapticSectionId
                                               << "of neuron "
                                               << synapse.postSynapticNeuronId);
        return;
    }

    
    if (postSynapticSegmentId >= nbPostSynapticSegments - 1)
    {
        PLUGIN_ERROR("Spine: " << neuronId << " / " << sectionId << " -> "
                               << synapse.postSynapticNeuronId << " / "
                               << synapse.postSynapticSectionId << " / "
                               << synapse.postSynapticSegmentId);
        PLUGIN_ERROR("Post-synaptic segment Id is out of range: "
                     << postSynapticSegmentId << "/" << nbPostSynapticSegments
                     << ". Section " << synapse.postSynapticSectionId
                     << " of neuron " << synapse.postSynapticNeuronId);
        return;
    }

    const auto& postSynapticSection = postSynapticSections.begin()->second;
    auto postSynapticSegmentId = synapse.postSynapticSegmentId;
    const auto nbPostSynapticSegments = postSynapticSection.points.size();

    const auto spineSmallRadius = radius * spineRadiusRatio * 0.15;
    const auto spineBaseRadius = radius * spineRadiusRatio * 0.25;
    const auto spineLargeRadius = radius * spineRadiusRatio;

    const Vector3d postSynapticSegmentDirection =
        normalize(postSynapticSection.points[postSynapticSegmentId + 1] -
                  postSynapticSection.points[postSynapticSegmentId]);

    const Vector3d postSynapticSurfacePosition =
        Vector3d(postSynapticSection.points[postSynapticSegmentId]) +
        postSynapticSegmentDirection * synapse.postSynapticSegmentDistance;

    const Vector3d animatedPostSynapticSurfacePosition =
        _animatedPosition(Vector4d(postSynapticSurfacePosition,
                                   spineBaseRadius),
                          synapse.postSynapticNeuronId);
    const auto direction =
        animatedPostSynapticSurfacePosition - preSynapticSurfacePosition;
    const auto l = length(direction) - spineLargeRadius;
#else
    const auto spineSmallRadius = radius * spineRadiusRatio * 0.5;
    const auto spineBaseRadius = radius * spineRadiusRatio * 0.75;
    const auto spineLargeRadius = radius * spineRadiusRatio * 2.5;

    const auto direction =
        Vector3d((rand() % 200 - 100) / 100.0, (rand() % 200 - 100) / 100.0,
                 (rand() % 200 - 100) / 100.0);
    const auto l = 6.f * radius;
#endif

    // container.addSphere(preSynapticSurfacePosition, DEFAULT_SPINE_RADIUS
    // * 3.f,
    //                     SpineMaterialId, neuronId);

    const auto origin = preSynapticSurfacePosition;
    const auto target = origin + normalize(direction) * l;

    // Create random shape between origin and target
    auto middle = (target + origin) / 2.0;
    const double d = length(target - origin) / 1.5;
    const auto i = neuronId * 4;
    middle += Vector3f(d * rnd2(i), d * rnd2(i + 1), d * rnd2(i + 2));
    const float spineMiddleRadius = spineSmallRadius + d * 0.1 * rnd2(i + 3);

    const auto displacement =
        Vector3f(spineDisplacementStrength, spineDisplacementFrequency, 0.f);
    Neighbours neighbours;
    if (!_details.useSdf)
        container.addSphere(target, spineLargeRadius, SpineMaterialId,
                            neuronId);
    neighbours.insert(container.addSphere(middle, spineMiddleRadius,
                                          SpineMaterialId, neuronId, neighbours,
                                          displacement));
    if (middle != origin)
        container.addCone(origin, spineSmallRadius, middle, spineMiddleRadius,
                          SpineMaterialId, neuronId, neighbours, displacement);
    if (middle != target)
        container.addCone(middle, spineMiddleRadius, target, spineLargeRadius,
                          SpineMaterialId, neuronId, neighbours, displacement);

    ++_nbSpines;
}

#if 0
void Neurons::_addSpine2(ThreadSafeContainer& container,
                         const uint64_t neuronId, const uint64_t morphologyId,
                         const uint64_t sectionId, const Synapse& synapse,
                         const size_t SpineMaterialId,
                         const Vector3d& preSynapticSurfacePosition)
{
    // TO REMOVE
    container.addSphere(preSynapticSurfacePosition, DEFAULT_SPINE_RADIUS * 3.f,
                        SpineMaterialId, neuronId);
    // TO REMOVE

    const auto& connector = DBConnector::getInstance();
    const double radius = DEFAULT_SPINE_RADIUS;

    // Spine geometry
    const auto spineSmallRadius = radius * spineRadiusRatio * 0.15;
    const auto spineBaseRadius = radius * spineRadiusRatio * 0.25;
    const auto spineLargeRadius = radius * spineRadiusRatio;

    const auto postSynapticSections = connector.getNeuronSections(
        _details.populationName, synapse.postSynapticNeuronId,
        "s.section_guid=" + std::to_string(synapse.postSynapticSectionId));

    if (postSynapticSections.empty())
    {
        PLUGIN_ERROR("Spine: " << neuronId << " / " << sectionId << " -> "
                               << synapse.postSynapticNeuronId << " / "
                               << synapse.postSynapticSectionId << " / "
                               << synapse.postSynapticSegmentId);
        PLUGIN_ERROR("Could not find section " << synapse.postSynapticSectionId
                                               << "of neuron "
                                               << synapse.postSynapticNeuronId);
        return;
    }

    const auto& postSynapticSection = postSynapticSections.begin()->second;
    auto postSynapticSegmentId = synapse.postSynapticSegmentId;
    const auto nbPostSynapticSegments = postSynapticSection.points.size();
    if (postSynapticSegmentId >= nbPostSynapticSegments - 1)
    {
        PLUGIN_ERROR("Spine: " << neuronId << " / " << sectionId << " -> "
                               << synapse.postSynapticNeuronId << " / "
                               << synapse.postSynapticSectionId << " / "
                               << synapse.postSynapticSegmentId);
        PLUGIN_ERROR("Post-synaptic segment Id is out of range: "
                     << postSynapticSegmentId << "/" << nbPostSynapticSegments
                     << ". Section " << synapse.postSynapticSectionId
                     << " of neuron " << synapse.postSynapticNeuronId);
        return;
    }

    // const Vector3d postSynapticSegmentDirection =
    //     normalize(postSynapticSection.points[postSynapticSegmentId + 1] -
    //               postSynapticSection.points[postSynapticSegmentId]);

    // const Vector3d postSynapticSurfacePosition =
    //     Vector3d(postSynapticSection.points[postSynapticSegmentId]) +
    //     postSynapticSegmentDirection * synapse.postSynapticSegmentDistance;

    // const Vector3d animatedPostSynapticSurfacePosition =
    //     _animatedPosition(Vector4d(postSynapticSurfacePosition,
    //                                spineBaseRadius),
    //                       synapse.postSynapticNeuronId);

    PLUGIN_ERROR("Postsynaptic neuron ID: " << synapse.postSynapticNeuronId);
    const auto postSynapticNeuronSomas =
        connector.getNeurons(_details.populationName,
                             "guid=" +
                                 std::to_string(synapse.postSynapticNeuronId));
    const auto& postSynapticSoma = postSynapticNeuronSomas.begin()->second;

    const Vector3f postSynapticSurfacePosition = _animatedPosition(
        Vector4d(postSynapticSoma.position +
                     postSynapticSoma.rotation *
                         Vector3d(
                             postSynapticSection.points[postSynapticSegmentId]),
                 DEFAULT_SPINE_RADIUS * 3.f),
        synapse.postSynapticNeuronId);

    container.addSphere(postSynapticSurfacePosition, DEFAULT_SPINE_RADIUS * 3.f,
                        SpineMaterialId, neuronId);
    container.addCone(preSynapticSurfacePosition, DEFAULT_SPINE_RADIUS * 3.f,
                      postSynapticSurfacePosition, DEFAULT_SPINE_RADIUS * 3.f,
                      SpineMaterialId, neuronId);
    // TO REMOVE

    // const auto direction =
    //     animatedPostSynapticSurfacePosition - preSynapticSurfacePosition;
    // const auto l = length(direction) - spineLargeRadius;

    // const auto origin = postSynapticSurfacePosition;
    // const auto target = origin + normalize(direction) * l;

    // // Create random shape between origin and target
    // auto middle = (target + origin) / 2.0;
    // const double d = length(target - origin) / 1.5;
    // const auto i = neuronId * 4;
    // middle += Vector3f(d * rnd2(i), d * rnd2(i + 1), d * rnd2(i + 2));
    // const float spineMiddleRadius = spineSmallRadius + d * 0.1 * rnd2(i + 3);

    // const auto displacement =
    //     Vector3f(spineDisplacementStrength, spineDisplacementFrequency, 0.f);
    // Neighbours neighbours;
    // if (!_details.useSdf)
    //     container.addSphere(target, spineLargeRadius, SpineMaterialId,
    //                         neuronId);
    // neighbours.insert(container.addSphere(middle, spineMiddleRadius,
    //                                       SpineMaterialId, neuronId,
    //                                       neighbours, displacement));
    // if (middle != origin)
    //     container.addCone(origin, spineSmallRadius, middle,
    //     spineMiddleRadius,
    //                       SpineMaterialId, neuronId, neighbours,
    //                       displacement);
    // if (middle != target)
    //     container.addCone(middle, spineMiddleRadius, target,
    //     spineLargeRadius,
    //                       SpineMaterialId, neuronId, neighbours,
    //                       displacement);
}
#endif

Vector4ds Neurons::getNeuronSectionPoints(const uint64_t neuronId,
                                          const uint64_t sectionId)
{
    const auto& connector = DBConnector::getInstance();
    const auto neurons =
        connector.getNeurons(_details.populationName,
                             "guid=" + std::to_string(neuronId));

    if (neurons.empty())
        PLUGIN_THROW("Neuron " + std::to_string(neuronId) + " does not exist");
    const auto& neuron = neurons.begin()->second;
    const auto sections =
        connector.getNeuronSections(_details.populationName, neuronId);

    if (sections.empty())
        PLUGIN_THROW("Section " + std::to_string(sectionId) +
                     " does not exist for neuron " + std::to_string(neuronId));
    const auto section = sections.begin()->second;
    Vector4ds points;
    for (const auto& point : section.points)
    {
        const Vector3d position =
            _scale * (neuron.position + neuron.rotation * Vector3d(point));
        const double radius = point.w;
        points.push_back({position.x, position.y, position.z, radius});
    }
    return points;
}

Vector3ds Neurons::getNeuronVaricosities(const uint64_t neuronId)
{
    if (_varicosities.find(neuronId) == _varicosities.end())
        PLUGIN_THROW("Neuron " + std::to_string(neuronId) + " does not exist");
    return _varicosities[neuronId];
}

std::string Neurons::_attachSimulationReport(Model& model)
{
    // Simulation report
    std::string sqlNodeFilter = _details.sqlNodeFilter;
    if (_details.simulationReportId != -1)
    {
        const auto& connector = DBConnector::getInstance();
        _reportType =
            connector.getNeuronReportType(_details.populationName,
                                          _details.simulationReportId);
        switch (_reportType)
        {
        case ReportType::undefined:
            PLUGIN_DEBUG("No report attached to the geometry");
            break;
        case ReportType::spike:
        {
            PLUGIN_INFO(1,
                        "Initialize spike simulation handler and restrain "
                        "guids to the simulated ones");
            auto handler = std::make_shared<SpikeSimulationHandler>(
                _details.populationName, _details.simulationReportId);
            model.setSimulationHandler(handler);
            setDefaultTransferFunction(model, DEFAULT_SIMULATION_VALUE_RANGE);
            if (!sqlNodeFilter.empty())
                sqlNodeFilter += "AND ";
            sqlNodeFilter += "guid IN (SELECT DISTINCT(node_guid) FROM " +
                             _details.populationName +
                             ".spike_report WHERE report_guid=" +
                             std::to_string(_details.simulationReportId) + ")";
            break;
        }
        case ReportType::soma:
        {
            PLUGIN_INFO(1,
                        "Initialize soma simulation handler and restrain guids "
                        "to the simulated ones");
            auto handler = std::make_shared<SomaSimulationHandler>(
                _details.populationName, _details.simulationReportId);
            model.setSimulationHandler(handler);
            setDefaultTransferFunction(model, DEFAULT_SIMULATION_VALUE_RANGE);
            if (!sqlNodeFilter.empty())
                sqlNodeFilter += "AND ";
            sqlNodeFilter += "guid IN (SELECT DISTINCT(node_guid) FROM " +
                             _details.populationName +
                             ".soma_report WHERE report_guid=" +
                             std::to_string(_details.simulationReportId) + ")";
            break;
        }
        case ReportType::compartment:
        {
            PLUGIN_INFO(
                1,
                "Initialize compartment simulation handler and restrain "
                "guids to the simulated ones");
            auto handler = std::make_shared<CompartmentSimulationHandler>(
                _details.populationName, _details.simulationReportId);
            model.setSimulationHandler(handler);
            setDefaultTransferFunction(model, DEFAULT_SIMULATION_VALUE_RANGE);
            if (!sqlNodeFilter.empty())
                sqlNodeFilter += "AND ";
            sqlNodeFilter += "guid IN (SELECT DISTINCT(node_guid) FROM " +
                             _details.populationName +
                             ".compartment_report WHERE report_guid=" +
                             std::to_string(_details.simulationReportId) + ")";
            break;
        }
        }
    }
    return sqlNodeFilter;
}

} // namespace morphology
} // namespace bioexplorer
