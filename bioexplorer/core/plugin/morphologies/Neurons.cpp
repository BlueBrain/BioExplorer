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

#include "Neurons.h"

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

// Mitochondria density per layer
const doubles MITOCHONDRIA_DENSITY = {0.0459, 0.0522, 0.064,
                                      0.0774, 0.0575, 0.0403};

Neurons::Neurons(Scene& scene, const NeuronsDetails& details)
    : Morphologies(details.radiusMultiplier, doublesToVector3d(details.scale))
    , _details(details)
    , _scene(scene)
{
    _radiusMultiplier =
        _details.radiusMultiplier > 0.0 ? _details.radiusMultiplier : 1.0;

    Timer chrono;
    _buildNeurons();
    PLUGIN_TIMER(chrono.elapsed(), "Neurons loaded");
}

void Neurons::_buildNeurons()
{
    const auto animationParams =
        doublesToAnimationDetails(_details.animationParams);
    srand(animationParams.seed);

    const auto& connector = DBConnector::getInstance();

    auto model = _scene.createModel();
    const auto useSdf = ((_details.morphologyRepresentation ==
                              MorphologyRepresentation::graph ||
                          _details.morphologyRepresentation ==
                              MorphologyRepresentation::orientation)
                             ? false
                             : _details.useSdf);
    const auto somas =
        connector.getNeurons(_details.populationName, _details.sqlNodeFilter);

    PLUGIN_INFO(1, "Building " << somas.size() << " neurons");

    // Neurons
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

        uint64_t index;
#pragma omp parallel for num_threads(nbDBConnections)
        for (index = 0; index < somas.size(); ++index)
        {
            if (omp_get_thread_num() == 0)
                PLUGIN_PROGRESS("Loading neurons", index,
                                somas.size() / nbDBConnections);

            auto it = somas.begin();
            std::advance(it, index);
            const auto& soma = it->second;
            const auto neuronId = it->first;

            ThreadSafeContainer container(*model, useSdf, _scale);
            _buildMorphology(container, soma, neuronId);

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

void Neurons::_buildSomasOnly(ThreadSafeContainer& container,
                              const NeuronSomaMap& somas,
                              const size_t baseMaterialId)
{
    uint64_t progress = 0;
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
                                somaMaterialId, NO_USER_DATA, {},
                                Vector3f(neuronSomaDisplacementStrength,
                                         neuronSomaDisplacementFrequency, 0.f));
        if (_details.generateInternals)
        {
            const double mitochondriaDensity =
                (soma.second.layer < MITOCHONDRIA_DENSITY.size()
                     ? MITOCHONDRIA_DENSITY[soma.second.layer]
                     : 0.0);

            _addSomaInternals(soma.first, container, baseMaterialId,
                              soma.second.position, _details.radiusMultiplier,
                              mitochondriaDensity);
        }
        ++progress;
    }
}

void Neurons::_buildOrientations(ThreadSafeContainer& container,
                                 const NeuronSomaMap& somas,
                                 const size_t baseMaterialId)
{
    const auto radius = _details.radiusMultiplier;
    const auto src = Vector4d(0, 0, 0, radius);
    uint64_t progress = 0;
    for (const auto soma : somas)
    {
        PLUGIN_PROGRESS("Loading soma orientations", progress, somas.size());
        const auto rotation = soma.second.rotation;
        const auto dst = Vector4d(rotation * (UP_VECTOR * radius *
                                              DEFAULT_ARROW_RADIUS_RATIO),
                                  radius);
        _addArrow(container, soma.second.position, rotation, src, dst,
                  NeuronSectionType::soma, 0);
        ++progress;
    }
}

void Neurons::_buildMorphology(ThreadSafeContainer& container,
                               const NeuronSoma& soma, const uint64_t neuronId)
{
    const auto& connector = DBConnector::getInstance();

    const auto& somaPosition = soma.position;
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
                const auto& point = section.second.points[0];
                somaRadius += 0.5 * length(Vector3d(point));
                ++count;
            }
        if (count > 0)
            somaRadius = _radiusMultiplier * somaRadius / count;
    }

    size_t baseMaterialId = 0;
    switch (_details.populationColorScheme)
    {
    case PopulationColorScheme::id:
        baseMaterialId = neuronId * NB_MATERIALS_PER_MORPHOLOGY;
        break;
    }
    size_t somaMaterialId = baseMaterialId;
    switch (_details.morphologyColorScheme)
    {
    case MorphologyColorScheme::section_type:
        somaMaterialId = baseMaterialId + MATERIAL_OFFSET_SOMA;
        break;
    case MorphologyColorScheme::section_orientation:
        somaMaterialId = getMaterialIdFromOrientation({1.0, 1.0, 1.0});
        break;
    }

    // Soma
    uint64_t somaGeometryIndex = 0;
    if (_details.loadSomas)
    {
        if (_details.showMembrane)
            somaGeometryIndex =
                container.addSphere(somaPosition, somaRadius, somaMaterialId,
                                    NO_USER_DATA, {},
                                    Vector3f(neuronSomaDisplacementStrength,
                                             neuronSomaDisplacementFrequency,
                                             0.f));
        if (_details.generateInternals)
            _addSomaInternals(neuronId, container, baseMaterialId, somaPosition,
                              somaRadius, mitochondriaDensity);
    }

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
            _addArrow(container, somaPosition, somaRotation,
                      section.second.points[0],
                      section.second.points[section.second.points.size() - 1],
                      sectionType, baseMaterialId);
            continue;
        }

        if (_details.showMembrane && _details.loadSomas &&
            section.second.parentId == SOMA_AS_PARENT)
        {
            const Vector4d somaPoint{somaPosition.x, somaPosition.y,
                                     somaPosition.z, somaRadius};
            const auto& point = section.second.points[0];

            // Section connected to the soma
            const auto srcRadius = somaPoint.w * 0.75 * _radiusMultiplier;
            const auto dstRadius = point.w * 0.5 * _radiusMultiplier;

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
            geometryIndex =
                container.addCone(Vector3d(somaPoint), srcRadius,
                                  somaPosition + somaRotation * Vector3d(point),
                                  dstRadius, somaMaterialId, NO_USER_DATA,
                                  neighbours,
                                  Vector3f(neuronSomaDisplacementStrength,
                                           neuronSomaDisplacementFrequency,
                                           0.f));
            neighbours.insert(geometryIndex);
        }

        _addSection(container, neuronId, section.first, section.second,
                    geometryIndex, somaPosition, somaRotation, somaRadius,
                    baseMaterialId, mitochondriaDensity);
    }

    // Synapses
    if (_details.loadSynapses)
        _addSpines(container, neuronId, somaPosition, somaRadius,
                   baseMaterialId);
}

void Neurons::_addVaricosity(Vector4fs& points)
{
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

void Neurons::_addArrow(ThreadSafeContainer& container,
                        const Vector3d& somaPosition,
                        const Quaterniond& somaRotation,
                        const Vector4d& srcNode, const Vector4d& dstNode,
                        const NeuronSectionType sectionType,
                        const size_t baseMaterialId)
{
    size_t sectionMaterialId = baseMaterialId;
    switch (_details.morphologyColorScheme)
    {
    case MorphologyColorScheme::section_type:
        sectionMaterialId = baseMaterialId + static_cast<size_t>(sectionType);
        break;
    case MorphologyColorScheme::section_orientation:
        sectionMaterialId =
            getMaterialIdFromOrientation(Vector3d(dstNode) - Vector3d(srcNode));
        break;
    }

    const auto src = somaPosition + somaRotation * Vector3d(srcNode);
    const auto dst = somaPosition + somaRotation * Vector3d(dstNode);
    const auto userData = 0;
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
                          const uint64_t neuronId, const uint64_t sectionId,
                          const Section& section,
                          const uint64_t somaGeometryIndex,
                          const Vector3d& somaPosition,
                          const Quaterniond& somaRotation,
                          const double somaRadius, const size_t baseMaterialId,
                          const double mitochondriaDensity)
{
    const auto sectionType = static_cast<NeuronSectionType>(section.type);
    auto useSdf = _details.useSdf;

    const auto& points = section.points;
    size_t sectionMaterialId = baseMaterialId;
    switch (_details.morphologyColorScheme)
    {
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
    for (uint64_t i = 0; i < localPoints.size() - 1; ++i)
    {
        const auto& srcPoint = localPoints[i];
        const Vector3d src = somaPosition + somaRotation * Vector3d(srcPoint);
        const double srcRadius = srcPoint.w * 0.5 * _radiusMultiplier;

        const auto& dstPoint = localPoints[i + 1];
        const Vector3d dst = somaPosition + somaRotation * Vector3d(dstPoint);
        const double dstRadius = dstPoint.w * 0.5 * _radiusMultiplier;
        const double sampleLength = length(dstPoint - srcPoint);
        sectionLength += sampleLength;

        if (_details.showMembrane)
        {
            if (i > 0 && _details.morphologyRepresentation !=
                             MorphologyRepresentation::segment)
                neighbours = {geometryIndex};

            Vector3f displacement{sectionDisplacementStrength,
                                  sectionDisplacementFrequency, 0.f};
            size_t materialId = sectionMaterialId;
            if (addVaricosity && _details.morphologyColorScheme ==
                                     MorphologyColorScheme::section_type)
            {
                if (i > middlePointIndex && i < middlePointIndex + 3)
                {
                    materialId = baseMaterialId + MATERIAL_OFFSET_VARICOSITY;
                    displacement = Vector3f(2 * sectionDisplacementStrength,
                                            sectionDisplacementFrequency, 0.f);
                }
                if (i == middlePointIndex + 1 || i == middlePointIndex + 3)
                    neighbours = {};
                if (i == middlePointIndex + 1)
                    _varicosities[neuronId].push_back(dst);
            }

            if (!useSdf)
                container.addSphere(dst, dstRadius, materialId, NO_USER_DATA);

            geometryIndex =
                container.addCone(src, srcRadius, dst, dstRadius, materialId,
                                  NO_USER_DATA, neighbours, displacement);

            neighbours.insert(geometryIndex);
        }
        sectionVolume += coneVolume(sampleLength, srcRadius, dstRadius);

        _bounds.merge(srcPoint);
    }

    if (sectionType == NeuronSectionType::axon)
    {
        if (_details.generateInternals)
            _addSectionInternals(container, somaPosition, somaRotation,
                                 sectionLength, sectionVolume, points,
                                 mitochondriaDensity, baseMaterialId);

        if (_details.generateExternals)
            _addAxonMyelinSheath(container, somaPosition, somaRotation,
                                 sectionLength, points, mitochondriaDensity,
                                 baseMaterialId);
    }
}

void Neurons::_addSectionInternals(
    ThreadSafeContainer& container, const Vector3d& somaPosition,
    const Quaterniond& somaRotation, const double sectionLength,
    const double sectionVolume, const Vector4fs& points,
    const double mitochondriaDensity, const size_t baseMaterialId)
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
                                        radius, mitochondrionMaterialId, -1);

                if (mitochondrionSegment > 0)
                {
                    Neighbours neighbours = {};
                    if (mitochondrionSegment > 1)
                        neighbours = {geometryIndex};
                    geometryIndex = container.addCone(
                        somaPosition + somaRotation * position, radius,
                        somaPosition + somaRotation * previousPosition,
                        previousRadius, mitochondrionMaterialId, -1, neighbours,
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

void Neurons::_addAxonMyelinSheath(ThreadSafeContainer& container,
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
        const auto srcRadius =
            srcPoint.w * 0.5 * myelinSteathRadiusRatio * _radiusMultiplier;

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
            const Vector3d dstPosition =
                somaPosition + somaRotation * Vector3d(dstPoint);
            const auto dstRadius =
                dstPoint.w * 0.5 * myelinSteathRadiusRatio * _radiusMultiplier;

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

void Neurons::_addSpines(ThreadSafeContainer& container,
                         const uint64_t somaIndex, const Vector3d somaPosition,
                         const double somaRadius, const size_t baseMaterialId)
{
    const auto& connector = DBConnector::getInstance();
    const auto synapses =
        connector.getNeuronSynapses(_details.populationName, somaIndex);
    const size_t spineMaterialId =
        baseMaterialId + MATERIAL_OFFSET_AFFERENT_SYNPASE;

    for (const auto& synapse : synapses)
        // TODO: Do not create spines on the soma, the data is not yet
        // acceptable
        if (length(synapse.second.surfacePosition - somaPosition) >
                somaRadius * 3.0 &&
            length(synapse.second.centerPosition - somaPosition) >
                somaRadius * 3.0)
            _addSpine(container, synapse.first, synapse.second,
                      spineMaterialId);
}

void Neurons::_addSpine(ThreadSafeContainer& container, const uint64_t guid,
                        const Synapse& synapse, const size_t SpineMaterialId)
{
    const double radius = DEFAULT_SPINE_RADIUS;

    // Spine geometry
    const double spineSmallRadius = radius * 0.3;
    const double spineBaseRadius = radius * 0.5;

    const auto direction = (synapse.centerPosition - synapse.surfacePosition);

    const auto surfaceOrigin = synapse.surfacePosition;
    const auto surfaceTarget = surfaceOrigin + direction;

    const auto spineLargeRadius = radius * spineRadiusRatio;

    // Create random shape between origin and target
    auto middle = (surfaceTarget + surfaceOrigin) / 2.0;
    const double d = length(surfaceTarget - surfaceOrigin) / 2.5;
    const auto i = guid * 4;
    middle += Vector3f(d * Shape::rnd2(i), d * Shape::rnd2(i + 1),
                       d * Shape::rnd2(i + 2));
    const float spineMiddleRadius =
        spineSmallRadius + d * 0.1 * Shape::rnd2(i + 3);

    const auto smallDisplacement =
        Vector3f(spineSmallRadius * spineDisplacementStrength * 0.5,
                 spineSmallRadius * spineDisplacementFrequency, 0.f);
    const auto largeDisplacement =
        Vector3f(spineSmallRadius * spineDisplacementStrength,
                 spineSmallRadius * spineDisplacementFrequency, 0.f);
    Neighbours neighbours;
    neighbours.insert(container.addSphere(surfaceOrigin, spineLargeRadius,
                                          SpineMaterialId, NO_USER_DATA,
                                          neighbours, largeDisplacement));
    neighbours.insert(container.addSphere(middle, spineMiddleRadius,
                                          SpineMaterialId, NO_USER_DATA,
                                          neighbours, largeDisplacement));
    if (surfaceOrigin != middle)
        container.addCone(surfaceOrigin, spineLargeRadius, middle,
                          spineMiddleRadius, SpineMaterialId, NO_USER_DATA,
                          neighbours, smallDisplacement);
    if (middle != surfaceTarget)
        container.addCone(middle, spineMiddleRadius, surfaceTarget,
                          spineSmallRadius, SpineMaterialId, NO_USER_DATA,
                          neighbours, smallDisplacement);
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

Vector3ds Neurons::getNeuronVaricosities(const uint64_t neuronId)
{
    if (_varicosities.find(neuronId) == _varicosities.end())
        PLUGIN_THROW("Neuron " + std::to_string(neuronId) + " does not exist");
    return _varicosities[neuronId];
}

} // namespace morphology
} // namespace bioexplorer
