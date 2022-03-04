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

const size_t NB_MATERIALS_PER_NEURON = 10;
const size_t MATERIAL_OFFSET_SOMA = 0;
const size_t MATERIAL_OFFSET_AXON = 1;
const size_t MATERIAL_OFFSET_DENDRITE = 2;
const size_t MATERIAL_OFFSET_APICAL_DENDRITE = 3;
const size_t MATERIAL_OFFSET_AFFERENT_SYNPASE = 4;
const size_t MATERIAL_OFFSET_EFFERENT_SYNPASE = 5;
const size_t MATERIAL_OFFSET_MITOCHONDRION = 6;
const size_t MATERIAL_OFFSET_NUCLEUS = 7;
const size_t MATERIAL_OFFSET_MYELIN_SHEATH = 8;

const int64_t NO_USER_DATA = -1;
const int64_t SOMA_AS_PARENT = -1;
const uint64_t NB_MYELIN_FREE_SEGMENTS = 4;

// Mitochondria density per layer
doubles MITOCHONDRIA_DENSITY = {0.0459, 0.0522, 0.064, 0.0774, 0.0575, 0.0403};

Neurons::Neurons(Scene& scene, const NeuronsDetails& details)
    : _details(details)
    , _scene(scene)
{
    Timer chrono;
    _buildNeuron();
    PLUGIN_TIMER(chrono.elapsed(), "Neurons loaded");
}

void Neurons::_buildNeuron()
{
    auto& connector = DBConnector::getInstance();

    auto model = _scene.createModel();
    MaterialSet materialIds;
    SDFMorphologyData sdfMorphologyData;
    const auto useSdf = _details.useSdf;
    const auto somas = connector.getNeurons(_details.sqlNodeFilter);

    // Neurons
    size_t previousMaterialId = std::numeric_limits<size_t>::max();
    size_t baseMaterialId = 0;
    Vector3ui indexOffset;

    uint64_t sdfGroupId = 0;
    for (const auto& soma : somas)
    {
        const auto somaId = soma.first;
        const auto& somaPosition = soma.second.center;
        const auto somaRadius = _details.radiusMultiplier;
        const auto layer = soma.second.layer;
        const double mitochondriaDensity =
            (layer < MITOCHONDRIA_DENSITY.size() ? MITOCHONDRIA_DENSITY[layer]
                                                 : 0.0);

        PLUGIN_PROGRESS("Loading Neurons", soma.first, somas.size());
        switch (_details.populationColorScheme)
        {
        case PopulationColorScheme::id:
            baseMaterialId = somaId * NB_MATERIALS_PER_NEURON;
            break;
        default:
            baseMaterialId = static_cast<size_t>(NeuronSectionType::soma);
        }
        materialIds.insert(baseMaterialId);

        const uint64_t sdfSomaGroupId = sdfGroupId;
        if (_details.loadSomas)
        {
            _addStepSphereGeometry(useSdf, somaPosition, somaRadius,
                                   baseMaterialId, NO_USER_DATA, *model,
                                   sdfMorphologyData, sdfSomaGroupId);
            if (_details.generateInternals)
            {
                _addSomaInternals(somaId, *model, baseMaterialId, somaPosition,
                                  somaRadius, mitochondriaDensity,
                                  sdfMorphologyData, sdfGroupId);
                materialIds.insert(baseMaterialId + MATERIAL_OFFSET_NUCLEUS);
                materialIds.insert(baseMaterialId +
                                   MATERIAL_OFFSET_MITOCHONDRION);
            }
        }

        if (_details.loadBasalDendrites || _details.loadApicalDendrites ||
            _details.loadAxons)
        {
            const auto sections =
                connector.getNeuronSections(somaId, _details.sqlSectionFilter);
            for (const auto& section : sections)
                _addSection(*model, section.first, section.second, somaPosition,
                            somaRadius, sdfSomaGroupId, baseMaterialId,
                            mitochondriaDensity, sdfMorphologyData, sdfGroupId,
                            materialIds);
        }
    }

    for (const auto materialId : materialIds)
    {
        Vector3f color{1.f, 1.f, 1.f};
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
        {"Number of Neurons", std::to_string(somas.size())}};

    _modelDescriptor.reset(new brayns::ModelDescriptor(std::move(model),
                                                       _details.assemblyName,
                                                       metadata));
    if (_modelDescriptor)
        _scene.addModel(_modelDescriptor);
    else
        PLUGIN_THROW("Neurons model could not be created");
}

void Neurons::_addSection(Model& model, const uint64_t sectionId,
                          const Section& section, const Vector3d& somaPosition,
                          const double somaRadius,
                          const uint64_t sdfSomaGroupId,
                          const size_t baseMaterialId,
                          const double mitochondriaDensity,
                          SDFMorphologyData& sdfMorphologyData,
                          uint64_t& sdfGroupId, MaterialSet& materialIds)
{
    ++sdfGroupId;
    const auto useSdf = _details.useSdf;
    const auto sectionType = static_cast<NeuronSectionType>(section.type);

    if (sectionType == NeuronSectionType::axon && !_details.loadAxons)
        return;
    if (sectionType == NeuronSectionType::basal_dendrite &&
        !_details.loadBasalDendrites)
        return;
    if (sectionType == NeuronSectionType::apical_dendrite &&
        !_details.loadApicalDendrites)
        return;

    const size_t sectionMaterialId =
        baseMaterialId +
        (_details.morphologyColorScheme == MorphologyColorScheme::section
             ? section.type
             : 0);
    materialIds.insert(sectionMaterialId);

    const auto& points = section.points;
    if (section.parentId == SOMA_AS_PARENT)
    {
        // Section connection to the soma
        const auto& point = points[0];
        _addStepConeGeometry(useSdf, somaPosition, somaRadius,
                             somaPosition + Vector3d(point), point.w * 0.5,
                             baseMaterialId, NO_USER_DATA, model,
                             sdfMorphologyData, sdfSomaGroupId);
    }

    double sectionLength = 0;
    double sectionVolume = 0;
    for (uint64_t i = 0; i < points.size() - 1; ++i)
    {
        const auto& srcPoint = points[i];
        const auto src = somaPosition + Vector3d(srcPoint);
        const double srcRadius = srcPoint.w * 0.5;

        const Vector3d sectionDisplacementParams = {std::min(srcRadius, 0.05),
                                                    1.2, 2.0};

        _bounds.merge(srcPoint);

        const auto& dstPoint = points[i + 1];
        const auto dst = somaPosition + Vector3d(dstPoint);
        const double dstRadius = dstPoint.w * 0.5;
        const double sampleLength = length(dstPoint - srcPoint);
        sectionLength += sampleLength;

        _addStepSphereGeometry(useSdf, dst, dstRadius, sectionMaterialId,
                               NO_USER_DATA, model, sdfMorphologyData,
                               sdfGroupId);
        _addStepConeGeometry(useSdf, src, srcRadius, dst, dstRadius,
                             sectionMaterialId, NO_USER_DATA, model,
                             sdfMorphologyData, sdfGroupId);
        sectionVolume += coneVolume(sampleLength, srcRadius, dstRadius);
    }

    if (sectionType == NeuronSectionType::axon)
    {
        uint64_t groupId = sectionId + sdfGroupId;
        if (_details.generateInternals)
            _addSectionInternals(sectionLength, sectionVolume, points,
                                 mitochondriaDensity, baseMaterialId,
                                 sdfMorphologyData, groupId, model);

        if (_details.generateExternals)
        {
            _addAxonMyelinSheath(somaPosition, sectionLength, points,
                                 mitochondriaDensity, baseMaterialId,
                                 sdfMorphologyData, groupId, model);
            materialIds.insert(baseMaterialId + MATERIAL_OFFSET_MYELIN_SHEATH);
        }
        sdfGroupId = groupId;
    }
}

size_t Neurons::_getNbMitochondrionSegments() const
{
    return 2 + rand() % 18;
}

void Neurons::_addSomaInternals(const uint64_t index, Model& model,
                                const size_t materialId,
                                const Vector3d& somaPosition,
                                const double somaRadius,
                                const double mitochondriaDensity,
                                SDFMorphologyData& sdfMorphologyData,
                                uint64_t& sdfGroupId)
{
    const auto useSdf = _details.useSdf;
    const double mitochondrionRadiusRatio = 0.025;
    const double mitochondrionDisplacementRatio = 20.0;
    const double nucleusDisplacementRatio = 2.0;
    const double nucleusRadius =
        somaRadius * 0.7; // 70% of the volume of the soma;
    const double mitochondrionRadius =
        somaRadius * mitochondrionRadiusRatio; // 5% of the volume of the soma

    const double somaInnerRadius = nucleusRadius + mitochondrionRadius;
    const double somaOutterRadius = somaRadius * 0.9;
    const double availableVolumeForMitochondria =
        sphereVolume(somaRadius) * mitochondriaDensity;

    // Soma nucleus
    const size_t nucleusMaterialId = materialId + MATERIAL_OFFSET_NUCLEUS;
    _addStepSphereGeometry(useSdf, somaPosition, nucleusRadius,
                           nucleusMaterialId, NO_USER_DATA, model,
                           sdfMorphologyData, sdfGroupId,
                           nucleusDisplacementRatio);

    // Mitochondria
    const size_t mitochondrionMaterialId =
        materialId + MATERIAL_OFFSET_MITOCHONDRION;
    double mitochondriaVolume = 0.0;
    while (mitochondriaVolume < availableVolumeForMitochondria)
    {
        const size_t nbSegments = _getNbMitochondrionSegments();
        const auto pointsInSphere =
            getPointsInSphere(nbSegments, somaInnerRadius / somaRadius);
        double previousRadius = mitochondrionRadius;
        for (size_t i = 0; i < nbSegments; ++i)
        {
            // Mitochondrion geometry
            const double radius =
                (1.0 + (rand() % 500 / 1000.0)) * mitochondrionRadius;
            const auto p2 = somaPosition + somaOutterRadius * pointsInSphere[i];
            _addStepSphereGeometry(useSdf, p2, radius, mitochondrionMaterialId,
                                   NO_USER_DATA, model, sdfMorphologyData,
                                   sdfGroupId, mitochondrionDisplacementRatio);

            mitochondriaVolume += sphereVolume(radius);

            if (i > 0)
            {
                const auto p1 =
                    somaPosition + somaOutterRadius * pointsInSphere[i - 1];
                _addStepConeGeometry(useSdf, p1, previousRadius, p2, radius,
                                     mitochondrionMaterialId, NO_USER_DATA,
                                     model, sdfMorphologyData, sdfGroupId,
                                     mitochondrionDisplacementRatio);

                mitochondriaVolume +=
                    coneVolume(length(p2 - p1), previousRadius, radius);
            }
            previousRadius = radius;
        }
        if (useSdf)
            ++sdfGroupId;
    }
}

void Neurons::_addSectionInternals(const double sectionLength,
                                   const double sectionVolume,
                                   const Vector4fs& points,
                                   const double mitochondriaDensity,
                                   const size_t materialId,
                                   SDFMorphologyData& sdfMorphologyData,
                                   uint64_t& sdfGroupId, Model& model)
{
    const auto useSdf = _details.useSdf;

    // Add mitochondria (density is per section, not for the full axon)
    const double mitochondrionSegmentSize = 0.25;
    const double mitochondrionRadiusRatio = 0.25;

    const size_t nbMaxMitochondrionSegments =
        sectionLength / mitochondrionSegmentSize;
    const double indexRatio =
        double(points.size()) / double(nbMaxMitochondrionSegments);

    double mitochondriaVolume = 0.0;

    size_t nbSegments = _getNbMitochondrionSegments();
    int mitochondrionSegment =
        -(rand() % (1 + nbMaxMitochondrionSegments / 10));
    double previousRadius;
    Vector3d previousPosition;

    ++sdfGroupId;
    for (size_t step = 0; step < nbMaxMitochondrionSegments; ++step)
    {
        if (mitochondriaVolume < sectionVolume * mitochondriaDensity &&
            mitochondrionSegment >= 0 && mitochondrionSegment < nbSegments)
        {
            const size_t srcIndex = size_t(step * indexRatio);
            const size_t dstIndex = size_t(step * indexRatio) + 1;
            if (dstIndex < points.size())
            {
                const auto& srcSample = points[srcIndex];
                const auto& dstSample = points[dstIndex];
                const double srcRadius =
                    srcSample.w * _details.radiusMultiplier;
                const Vector3d srcPosition{
                    srcSample.x + srcRadius * (rand() % 100 - 50) / 500.0,
                    srcSample.y + srcRadius * (rand() % 100 - 50) / 500.0,
                    srcSample.z + srcRadius * (rand() % 100 - 50) / 500.0};
                const double dstRadius =
                    dstSample.w * _details.radiusMultiplier;
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
                    (1.0 + ((rand() % 500) / 1000.0)) * mitocondrionRadius;

                const size_t mitochondrionMaterialId =
                    materialId + MATERIAL_OFFSET_MITOCHONDRION;
                _addStepSphereGeometry(useSdf, position, radius,
                                       mitochondrionMaterialId, -1, model,
                                       sdfMorphologyData, sdfGroupId,
                                       mitochondrionRadiusRatio);
                mitochondriaVolume += sphereVolume(radius);

                if (mitochondrionSegment > 0)
                {
                    _addStepConeGeometry(useSdf, position, radius,
                                         previousPosition, previousRadius,
                                         mitochondrionMaterialId, -1, model,
                                         sdfMorphologyData, sdfGroupId,
                                         mitochondrionRadiusRatio);
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
            ++sdfGroupId;
        }
    }
}

void Neurons::_addAxonMyelinSheath(const Vector3d& somaPosition,
                                   const double sectionLength,
                                   const Vector4fs& points,
                                   const double mitochondriaDensity,
                                   const size_t materialId,
                                   SDFMorphologyData& sdfMorphologyData,
                                   uint64_t& sdfGroupId, Model& model)
{
    if (sectionLength == 0 || points.empty())
        return;

    const auto useSdf = _details.useSdf;

    const double myelinSteathLength = 10.0;
    const double myelinSteathRadius = 0.5;
    const double myelinSteathDisplacementRatio = 0.25;
    const size_t myelinSteathMaterialId =
        materialId + MATERIAL_OFFSET_MYELIN_SHEATH;

    if (sectionLength < 2 * myelinSteathLength)
        return;

    const uint64_t nbPoints = points.size();

    uint64_t i = NB_MYELIN_FREE_SEGMENTS; // Ignore first 3 segments
    while (i < nbPoints - NB_MYELIN_FREE_SEGMENTS)
    {
        // Start surrounding segments with myelin steaths
        ++sdfGroupId;
        const auto& srcPoint = points[i];
        const Vector3d srcPosition = somaPosition + Vector3d(srcPoint);
        if (!useSdf)
            _addStepSphereGeometry(useSdf, srcPosition, myelinSteathRadius,
                                   myelinSteathMaterialId, NO_USER_DATA, model,
                                   sdfMorphologyData, sdfGroupId);

        double currentLength = 0;
        Vector3d previousPosition = srcPosition;
        while (currentLength < myelinSteathLength &&
               i < nbPoints - NB_MYELIN_FREE_SEGMENTS)
        {
            ++i;
            const auto& dstPoint = points[i];
            const Vector3d dstPosition = somaPosition + Vector3d(dstPoint);
            currentLength += length(dstPosition - previousPosition);
            if (!useSdf)
                _addStepSphereGeometry(useSdf, dstPosition, myelinSteathRadius,
                                       myelinSteathMaterialId, NO_USER_DATA,
                                       model, sdfMorphologyData, sdfGroupId);
            _addStepConeGeometry(useSdf, dstPosition, myelinSteathRadius,
                                 previousPosition, myelinSteathRadius,
                                 myelinSteathMaterialId, NO_USER_DATA, model,
                                 sdfMorphologyData, sdfGroupId,
                                 myelinSteathDisplacementRatio);
            previousPosition = dstPosition;
        }
        i += NB_MYELIN_FREE_SEGMENTS; // Leave free segments between myelin
                                      // steaths
    }
}

} // namespace morphology
} // namespace bioexplorer
