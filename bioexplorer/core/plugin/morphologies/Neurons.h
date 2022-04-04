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
 * Load a population of neurons from the database according to specified
 * parameters
 */
class Neurons : public Morphologies
{
public:
    /**
     * @brief Construct a new Neurons object
     *
     * @param scene 3D scene into which neurons should be loaded
     * @param details Set of attributes defining how neurons should be loaded
     */
    Neurons(Scene& scene, const NeuronsDetails& details);

    /**
     * @brief Get the Neuron section 3D points for a given section Id
     *
     * @param neuronId Neuron identifier
     * @param sectionId Neuron section identifier
     * @return Vector4ds 3D points, including radius, for the specified section
     */
    Vector4ds getNeuronSectionPoints(const uint64_t neuronId,
                                     const uint64_t sectionId);

private:
    void _buildNeuron();

    void _addSection(ThreadSafeContainer& container, const uint64_t sectionId,
                     const Section& section, const size_t somaIdx,
                     const Vector3d& somaPosition,
                     const Quaterniond& somaRotation, const double somaRadius,
                     const size_t baseMaterialId,
                     const double mitochondriaDensity,
                     MaterialSet& materialIds);

    void _addSpine(ThreadSafeContainer& container, const Synapse& synapse,
                   const size_t baseMaterialId);

    void _addSpines(ThreadSafeContainer& container, const uint64_t somaIndex,
                    const Vector3d somaPosition, const double somaRadius,
                    const size_t baseMaterialId);

    void _addSectionInternals(
        ThreadSafeContainer& container, const Vector3d& somaPosition,
        const Quaterniond& somaRotation, const double sectionLength,
        const double sectionVolume, const Vector4fs& points,
        const double mitochondriaDensity, const size_t baseMaterialId);

    void _addAxonMyelinSheath(ThreadSafeContainer& container,
                              const Vector3d& somaPosition,
                              const Quaterniond& somaRotation,
                              const double sectionLength,
                              const Vector4fs& points,
                              const double mitochondriaDensity,
                              const size_t materialId);

    const NeuronsDetails _details;
    Scene& _scene;
};
} // namespace morphology
} // namespace bioexplorer
