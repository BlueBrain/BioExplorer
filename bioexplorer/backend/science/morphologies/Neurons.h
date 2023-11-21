/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#pragma once

#include "Morphologies.h"

#include <science/api/Params.h>
#include <science/common/Types.h>

namespace bioexplorer
{
namespace morphology
{
using Varicosities = std::map<uint64_t, Vector3ds>;

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
    Neurons(core::Scene& scene, const details::NeuronsDetails& details, const core::Vector3d& assemblyPosition,
            const core::Quaterniond& assemblyRotation, const core::LoaderProgress& callback = core::LoaderProgress());

    /**
     * @brief Get the neuron section 3D points for a given section Id
     *
     * @param neuronId Neuron identifier
     * @param sectionId Neuron section identifier
     * @return Vector4ds 3D points, including radius, for the specified section
     */
    Vector4ds getNeuronSectionPoints(const uint64_t neuronId, const uint64_t sectionId);

    /**
     * @brief Get the neuron varicosities location in space
     *
     * @param neuronId Neuron identifier
     * @return Vector3ds Varicosity locations
     */
    Vector3ds getNeuronVaricosities(const uint64_t neuronId);

private:
    double _getDisplacementValue(const DisplacementElement& element) final;

    void _logRealismParams();

    void _buildModel(const core::LoaderProgress& callback);

    void _buildContours(common::ThreadSafeContainer& container, const NeuronSomaMap& somas);

    void _buildSurface(const NeuronSomaMap& somas);

    void _buildSomasOnly(common::ThreadSafeContainer& container, const NeuronSomaMap& somas);

    void _buildOrientations(common::ThreadSafeContainer& container, const NeuronSomaMap& somas);

    void _buildMorphology(common::ThreadSafeContainer& container, const uint64_t neuronId, const NeuronSoma& soma,
                          const uint64_t neuronIndex);

    SectionSynapseMap _buildDebugSynapses(const uint64_t neuronId, const SectionMap& sections);

    void _addArrow(common::ThreadSafeContainer& container, const uint64_t neuronId, const core::Vector3d& somaPosition,
                   const core::Quaterniond& somaRotation, const core::Vector4d& srcNode, const core::Vector4d& dstNode,
                   const details::NeuronSectionType sectionType, const size_t baseMaterialId,
                   const double distanceToSoma);

    void _addSection(common::ThreadSafeContainer& container, const uint64_t neuronId, const uint64_t morphologyId,
                     const uint64_t sectionId, const Section& section, const uint64_t somaGeometryIndex,
                     const core::Vector3d& somaPosition, const core::Quaterniond& somaRotation, const double somaRadius,
                     const size_t baseMaterialId, const double mitochondriaDensity, const uint64_t somaUserData,
                     const SectionSynapseMap& synapses, const double distanceToSoma);

    void _addSpine(common::ThreadSafeContainer& container, const uint64_t neuronId, const uint64_t morphologyId,
                   const uint64_t sectionId, const Synapse& synapse, const size_t baseMaterialId,
                   const core::Vector3d& surfacePosition, const double radiusAtSurfacePosition);

    void _addSectionInternals(common::ThreadSafeContainer& container, const uint64_t neuronId,
                              const core::Vector3d& somaPosition, const core::Quaterniond& somaRotation,
                              const double sectionLength, const double sectionVolume, const core::Vector4fs& points,
                              const double mitochondriaDensity, const size_t baseMaterialId);

    void _addAxonMyelinSheath(common::ThreadSafeContainer& container, const uint64_t neuronId,
                              const core::Vector3d& somaPosition, const core::Quaterniond& somaRotation,
                              const double sectionLength, const core::Vector4fs& points,
                              const double mitochondriaDensity, const size_t materialId);

    void _addVaricosity(core::Vector4fs& points);

    void _attachSimulationReport(core::Model& model);

    const details::NeuronsDetails _details;
    core::Scene& _scene;
    Varicosities _varicosities;
    uint64_t _nbSpines{0};
    double _maxDistanceToSoma{0.0};
    core::Vector2d _minMaxSomaRadius{1e6, -1e6};
    common::SimulationReport _simulationReport;
};
} // namespace morphology
} // namespace bioexplorer
