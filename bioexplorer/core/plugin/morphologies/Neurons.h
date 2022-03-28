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

#pragma once

#include "Morphologies.h"

#include <plugin/api/Params.h>
#include <plugin/common/Types.h>

namespace bioexplorer
{
namespace morphology
{
using namespace brayns;
using namespace common;

/**
 * Load Neurons from database
 */
class Neurons : public Morphologies
{
public:
    Neurons(Scene& scene, const NeuronsDetails& details);

    Vector4ds getNeuronSectionPoints(const uint64_t neuronId,
                                     const uint64_t sectionId);

private:
    void _buildNeuron();

    void _addSection(Model& model, const uint64_t sectionId,
                     const Section& section, const size_t somaIdx,
                     const Vector3d& somaPosition,
                     const Quaterniond& somaRotation, const double somaRadius,
                     const size_t baseMaterialId,
                     const double mitochondriaDensity,
                     SDFMorphologyData& sdfMorphologyData,
                     MaterialSet& materialIds);

    void _addSpine(Model& model, const Synapse& synapse,
                   const size_t baseMaterialId,
                   SDFMorphologyData& sdfMorphologyData);

    void _addSpines(Model& model, const uint64_t somaIndex,
                    const Vector3d somaPosition, const double somaRadius,
                    const size_t baseMaterialId,
                    SDFMorphologyData& sdfMorphologyData);

    void _addSectionInternals(
        const Vector3d& somaPosition, const Quaterniond& somaRotation,
        const double sectionLength, const double sectionVolume,
        const Vector4fs& points, const double mitochondriaDensity,
        const size_t baseMaterialId, SDFMorphologyData& sdfMorphologyData,
        Model& model);

    void _addAxonMyelinSheath(
        const Vector3d& somaPosition, const Quaterniond& somaRotation,
        const double sectionLength, const Vector4fs& points,
        const double mitochondriaDensity, const size_t materialId,
        SDFMorphologyData& sdfMorphologyData, Model& model);

    const NeuronsDetails _details;
    Scene& _scene;
};
} // namespace morphology
} // namespace bioexplorer
