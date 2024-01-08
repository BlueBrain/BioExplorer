/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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

namespace bioexplorer
{
namespace morphology
{
/**
 * Load synapse efficacy information from database
 */
class Synapses : public Morphologies
{
public:
    /**
     * @brief Construct a new Synapses object
     *
     * @param scene 3D scene into which the white matter should be loaded
     * @param details Set of attributes defining how the synapse efficacy should
     * be loaded
     */
    Synapses(core::Scene& scene, const details::SynapsesDetails& details, const core::Vector3d& assemblyPosition,
             const core::Quaterniond& assemblyRotation);

private:
    double _getDisplacementValue(const DisplacementElement& element) final;

    void _buildModel();
    void _addSpine(common::ThreadSafeContainer& container, const uint64_t guid, const Synapse& synapse,
                   const size_t SpineMaterialId);

    const details::SynapsesDetails _details;
    core::Scene& _scene;
};
} // namespace morphology
} // namespace bioexplorer
