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

#include "Neurons.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>
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

const uint64_t NB_MYELIN_FREE_SEGMENTS = 4;
const double DEFAULT_SPINE_RADIUS = 0.25;

// Mitochondria density per layer
const doubles MITOCHONDRIA_DENSITY = {0.0459, 0.0522, 0.064,
                                      0.0774, 0.0575, 0.0403};

Neurons::Neurons(Scene& scene, const NeuronsDetails& details)
    : Morphologies(details.radiusMultiplier)
    , _details(details)
    , _scene(scene)
{
    _radiusMultiplier =
        _details.radiusMultiplier > 0.0 ? _details.radiusMultiplier : 1.0;

    Timer chrono;
    _buildNeuron();
    PLUGIN_TIMER(chrono.elapsed(), "Neurons loaded");
}

void Neurons::_buildNeuron()
{
    auto& connector = DBConnector::getInstance();

    auto model = _scene.createModel();
    MaterialSet materialIds;
    const auto useSdf = _details.useSdf;
    const auto somas =
        connector.getNeurons(_details.populationName, _details.sqlNodeFilter);

    PLUGIN_INFO(1, "Building " << somas.size() << " neurons");

    // Neurons
    size_t previousMaterialId = std::numeric_limits<size_t>::max();
    size_t baseMaterialId = 0;
    Vector3ui indexOffset;

    std::vector<ParallelModelContainer> containers;
    uint64_t index;
#pragma omp parallel for private(index)
    for (index = 0; index < somas.size(); ++index)
    {
        if (omp_get_thread_num() == 0)
            PLUGIN_PROGRESS("Loading Neurons", index,
                            somas.size() / omp_get_max_threads());

        auto it = somas.begin();
        std::advance(it, index);
        const auto& soma = it->second;
        const auto somaId = it->first;

        ParallelModelContainer modelContainer(*model, useSdf, _scale);

        const auto& somaPosition = soma.position;
        const auto& somaRotation = soma.rotation;
        const auto layer = soma.layer;
        const double mitochondriaDensity =
            (layer < MITOCHONDRIA_DENSITY.size() ? MITOCHONDRIA_DENSITY[layer]
                                                 : 0.0);

        // Soma radius
        double somaRadius = 0.0;
        const auto sections =
            connector.getNeuronSections(_details.populationName, somaId,
                                        _details.sqlSectionFilter);
        uint64_t count = 1;
        for (const auto& section : sections)
            if (section.second.parentId == SOMA_AS_PARENT)
            {
                const auto& point = section.second.points[0];
                somaRadius += 0.5 * length(Vector3d(point));
                ++count;
            }
        somaRadius = _radiusMultiplier * somaRadius / count;

        switch (_details.populationColorScheme)
        {
        case PopulationColorScheme::id:
            baseMaterialId = somaId * NB_MATERIALS_PER_MORPHOLOGY;
            break;
        }
        materialIds.insert(baseMaterialId);
        const auto somaMaterialId =
            baseMaterialId +
            (_details.morphologyColorScheme == MorphologyColorScheme::section
                 ? MATERIAL_OFFSET_SOMA
                 : 0);
        materialIds.insert(somaMaterialId);

        // Soma
        uint64_t somaGeometryIndex = 0;
        if (_details.loadSomas)
        {
            somaGeometryIndex =
                modelContainer.addSphere(somaPosition, somaRadius,
                                         somaMaterialId, NO_USER_DATA, {},
                                         DEFAULT_SOMA_DISPLACEMENT);
            if (_details.generateInternals)
            {
                _addSomaInternals(somaId, modelContainer, baseMaterialId,
                                  somaPosition, somaRadius,
                                  mitochondriaDensity);
                materialIds.insert(baseMaterialId + MATERIAL_OFFSET_NUCLEUS);
                materialIds.insert(baseMaterialId +
                                   MATERIAL_OFFSET_MITOCHONDRION);
            }
        }

        // Sections (dendrites and axon)
        if (_details.loadBasalDendrites || _details.loadApicalDendrites ||
            _details.loadAxon)
        {
            uint64_t geometryIndex = 0;
            Neighbours neighbours;
            neighbours.insert(somaGeometryIndex);

            for (const auto& section : sections)
            {
                if (_details.loadSomas &&
                    section.second.parentId == SOMA_AS_PARENT)
                {
                    const Vector4d somaPoint{somaPosition.x, somaPosition.y,
                                             somaPosition.z, somaRadius};
                    const auto& point = section.second.points[0];

                    // Section connected to the soma
                    geometryIndex = modelContainer.addCone(
                        Vector3d(somaPoint), somaPoint.w * 0.75f,
                        (somaPosition + somaRotation * Vector3d(point)),
                        point.w * 0.5 * _radiusMultiplier, somaMaterialId,
                        NO_USER_DATA, neighbours, DEFAULT_SOMA_DISPLACEMENT);
                    neighbours.insert(geometryIndex);
                }

                _addSection(modelContainer, section.first, section.second,
                            geometryIndex, somaPosition, somaRotation,
                            somaRadius, baseMaterialId, mitochondriaDensity,
                            materialIds);
            }
        }

        if (_details.loadSynapses)
        {
            _addSpines(modelContainer, somaId, somaPosition, somaRadius,
                       baseMaterialId);
#pragma omp critical
            materialIds.insert(baseMaterialId +
                               MATERIAL_OFFSET_AFFERENT_SYNPASE);
        }

#pragma omp critical
        containers.push_back(modelContainer);
    }

    for (size_t i = 0; i < containers.size(); ++i)
    {
        const float progress = 1.f + i;
        PLUGIN_PROGRESS("- Compiling 3D geometry...", progress,
                        containers.size());
        auto& container = containers[i];
        container.commitToModel();
    }

    PLUGIN_INFO(1, "Creating materials...");

    ModelMetadata metadata = {
        {"Number of Neurons", std::to_string(somas.size())}};

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

void Neurons::_addSection(ParallelModelContainer& modelContainer,
                          const uint64_t sectionId, const Section& section,
                          const size_t somaGeometryIndex,
                          const Vector3d& somaPosition,
                          const Quaterniond& somaRotation,
                          const double somaRadius, const size_t baseMaterialId,
                          const double mitochondriaDensity,
                          MaterialSet& materialIds)
{
    const auto sectionType = static_cast<NeuronSectionType>(section.type);
    auto useSdf = _details.useSdf;

    const size_t sectionMaterialId =
        baseMaterialId +
        (_details.morphologyColorScheme == MorphologyColorScheme::section
             ? section.type
             : 0);
    materialIds.insert(sectionMaterialId);

    const auto& points = section.points;
    if (sectionType == NeuronSectionType::axon && !_details.loadAxon)
        return;
    if (sectionType == NeuronSectionType::basal_dendrite &&
        !_details.loadBasalDendrites)
        return;
    if (sectionType == NeuronSectionType::apical_dendrite &&
        !_details.loadApicalDendrites)
        return;

    double sectionLength = 0.0;
    double sectionVolume = 0.0;
    uint64_t geometryIndex = 0;
    for (uint64_t i = 0; i < points.size() - 1; ++i)
    {
        const auto& srcPoint = points[i];
        const Vector3d src = somaPosition + somaRotation * Vector3d(srcPoint);
        const double srcRadius = srcPoint.w * 0.5 * _radiusMultiplier;

        const auto& dstPoint = points[i + 1];
        const Vector3d dst = somaPosition + somaRotation * Vector3d(dstPoint);
        const double dstRadius = dstPoint.w * 0.5 * _radiusMultiplier;
        const double sampleLength = length(dstPoint - srcPoint);
        sectionLength += sampleLength;

        if (!useSdf)
            modelContainer.addSphere(dst, dstRadius, sectionMaterialId,
                                     NO_USER_DATA, {});

        Neighbours neighbours{somaGeometryIndex};
        if (i > 0)
            neighbours = {geometryIndex};
        geometryIndex =
            modelContainer.addCone(src, srcRadius, dst, dstRadius,
                                   sectionMaterialId, NO_USER_DATA, neighbours,
                                   DEFAULT_SECTION_DISPLACEMENT);
        sectionVolume += coneVolume(sampleLength, srcRadius, dstRadius);

        _bounds.merge(srcPoint);
    }

    if (sectionType == NeuronSectionType::axon)
    {
        if (_details.generateInternals)
            _addSectionInternals(modelContainer, somaPosition, somaRotation,
                                 sectionLength, sectionVolume, points,
                                 mitochondriaDensity, baseMaterialId);

        if (_details.generateExternals)
        {
            _addAxonMyelinSheath(modelContainer, somaPosition, somaRotation,
                                 sectionLength, points, mitochondriaDensity,
                                 baseMaterialId);
            materialIds.insert(baseMaterialId + MATERIAL_OFFSET_MYELIN_SHEATH);
        }
    }
}

void Neurons::_addSectionInternals(
    ParallelModelContainer& modelContainer, const Vector3d& somaPosition,
    const Quaterniond& somaRotation, const double sectionLength,
    const double sectionVolume, const Vector4fs& points,
    const double mitochondriaDensity, const size_t baseMaterialId)
{
    if (mitochondriaDensity == 0.0)
        return;

    const auto useSdf = _details.useSdf;

    // Add mitochondria (density is per section, not for the full axon)
    const double mitochondrionSegmentSize = 0.25;
    const double mitochondrionRadiusRatio = 0.25;

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
    for (size_t step = 0; step < nbMaxMitochondrionSegments; ++step)
    {
        if (mitochondriaVolume < sectionVolume * mitochondriaDensity &&
            mitochondrionSegment >= 0 && mitochondrionSegment < nbSegments)
        {
            const uint64_t srcIndex = uint64_t(step * indexRatio);
            const uint64_t dstIndex = uint64_t(step * indexRatio) + 1;
            if (dstIndex < points.size())
            {
                const auto& srcSample = points[srcIndex];
                const auto& dstSample = points[dstIndex];
                const double srcRadius = srcSample.w * 0.5 * _radiusMultiplier;
                const Vector3d srcPosition{
                    srcSample.x + srcRadius * (rand() % 100 - 50) / 500.0,
                    srcSample.y + srcRadius * (rand() % 100 - 50) / 500.0,
                    srcSample.z + srcRadius * (rand() % 100 - 50) / 500.0};
                const double dstRadius = dstSample.w * 0.5 * _radiusMultiplier;
                const Vector3d dstPosition{
                    dstSample.x + dstRadius * (rand() % 100 - 50) / 500.0,
                    dstSample.y + dstRadius * (rand() % 100 - 50) / 500.0,
                    dstSample.z + dstRadius * (rand() % 100 - 50) / 500.0};

                const Vector3d direction = dstPosition - srcPosition;
                const Vector3d position =
                    srcPosition + direction * (step * indexRatio - srcIndex);
                const double mitocondrionRadius =
                    srcRadius * mitochondrionRadiusRatio;
                const double radius =
                    (1.0 + (rand() % 1000 - 500) / 1000.0) * mitocondrionRadius;

                Neighbours neighbours;
                if (mitochondrionSegment != 0)
                    neighbours = {geometryIndex};

                if (!useSdf)
                    modelContainer.addSphere(somaPosition +
                                                 somaRotation * position,
                                             radius, mitochondrionMaterialId,
                                             -1, {});

                if (mitochondrionSegment > 0)
                {
                    Neighbours neighbours = {};
                    if (mitochondrionSegment > 1)
                        neighbours = {geometryIndex};
                    geometryIndex = modelContainer.addCone(
                        somaPosition + somaRotation * position, radius,
                        somaPosition + somaRotation * previousPosition,
                        previousRadius, mitochondrionMaterialId, -1,
                        neighbours);

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
        }
    }
}

void Neurons::_addAxonMyelinSheath(ParallelModelContainer& modelContainer,
                                   const Vector3d& somaPosition,
                                   const Quaterniond& somaRotation,
                                   const double sectionLength,
                                   const Vector4fs& points,
                                   const double mitochondriaDensity,
                                   const size_t baseMaterialId)
{
    if (sectionLength == 0 || points.empty())
        return;

    const auto useSdf = _details.useSdf;

    const double myelinSteathLength = 10.0;
    const double myelinSteathRadius = 0.5;
    const double myelinSteathDisplacementRatio = 0.25;
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
        const Vector3d srcPosition =
            somaPosition + somaRotation * Vector3d(srcPoint);
        if (!useSdf)
            modelContainer.addSphere(srcPosition, myelinSteathRadius,
                                     myelinSteathMaterialId, NO_USER_DATA, {});

        double currentLength = 0;
        Vector3d previousPosition = srcPosition;
        while (currentLength < myelinSteathLength &&
               i < nbPoints - NB_MYELIN_FREE_SEGMENTS)
        {
            ++i;
            const auto& dstPoint = points[i];
            const Vector3d dstPosition =
                somaPosition + somaRotation * Vector3d(dstPoint);
            currentLength += length(dstPosition - previousPosition);
            if (!useSdf)
                modelContainer.addSphere(dstPosition, myelinSteathRadius,
                                         myelinSteathMaterialId, NO_USER_DATA,
                                         {});
            modelContainer.addCone(dstPosition, myelinSteathRadius,
                                   previousPosition, myelinSteathRadius,
                                   myelinSteathMaterialId, NO_USER_DATA, {},
                                   myelinSteathDisplacementRatio);
            previousPosition = dstPosition;
        }
        i += NB_MYELIN_FREE_SEGMENTS; // Leave free segments between myelin
                                      // steaths
    }
}

void Neurons::_addSpines(ParallelModelContainer& modelContainer,
                         const uint64_t somaIndex, const Vector3d somaPosition,
                         const double somaRadius, const size_t baseMaterialId)
{
    auto& connector = DBConnector::getInstance();
    const auto synapses =
        connector.getNeuronSynapses(_details.populationName, somaIndex);
    const size_t SpineMaterialId =
        baseMaterialId + MATERIAL_OFFSET_AFFERENT_SYNPASE;

    for (const auto& synapse : synapses)
        // TODO: Do not create spines on the soma, the data is not yet
        // acceptable
        if (length(synapse.second.surfacePosition - somaPosition) >
                somaRadius * 3.0 &&
            length(synapse.second.centerPosition - somaPosition) >
                somaRadius * 3.0)
            _addSpine(modelContainer, synapse.second, SpineMaterialId);
}

void Neurons::_addSpine(ParallelModelContainer& modelContainer,
                        const Synapse& synapse, const size_t SpineMaterialId)
{
    const double radius = DEFAULT_SPINE_RADIUS;

    // Spine geometry
    const double spineRadiusRatio = 1.5;
    const double spineSmallRadius = radius * 0.3;
    const double spineBaseRadius = radius * 0.5;
    const double spineDisplacementRatio = 2.0;

    const auto direction = (synapse.centerPosition - synapse.surfacePosition);

    const auto surfaceOrigin = synapse.surfacePosition;
    const auto surfaceTarget = surfaceOrigin + direction;

    const auto spineLargeRadius = radius * spineRadiusRatio;

    // Create random shape between origin and target
    auto middle = (surfaceTarget + surfaceOrigin) / 2.0;
    const double d = length(surfaceTarget - surfaceOrigin) / 5.0;
    middle +=
        Vector3f(d * (rand() % 1000 / 1000.0), d * (rand() % 1000 / 1000.0),
                 d * (rand() % 1000 / 1000.0));
    const float spineMiddleRadius =
        spineSmallRadius + d * 0.1 * (rand() % 1000 / 1000.0);
    const auto idx1 = modelContainer.addSphere(surfaceOrigin, spineLargeRadius,
                                               SpineMaterialId, NO_USER_DATA,
                                               {}, spineDisplacementRatio);
    const auto idx2 =
        modelContainer.addSphere(middle, spineMiddleRadius, SpineMaterialId,
                                 NO_USER_DATA, {idx1}, spineDisplacementRatio);
    if (surfaceOrigin != middle)
        modelContainer.addCone(surfaceOrigin, spineLargeRadius, middle,
                               spineMiddleRadius, SpineMaterialId, NO_USER_DATA,
                               {idx1, idx2}, spineDisplacementRatio);
    if (middle != surfaceTarget)
        modelContainer.addCone(middle, spineMiddleRadius, surfaceTarget,
                               spineSmallRadius, SpineMaterialId, NO_USER_DATA,
                               {idx1, idx2}, spineDisplacementRatio);
}

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

} // namespace morphology
} // namespace bioexplorer
