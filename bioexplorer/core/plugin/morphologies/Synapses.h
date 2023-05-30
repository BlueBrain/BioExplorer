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

#pragma once

#include "Morphologies.h"

namespace bioexplorer
{
namespace morphology
{
using namespace brayns;
using namespace common;
using namespace details;

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
    Synapses(Scene& scene, const SynapsesDetails& details, const Vector3d& assemblyPosition,
             const Quaterniond& assemblyRotation);

private:
    double _getDisplacementValue(const DisplacementElement& element) final;

    void _buildModel();
    void _addSpine(ThreadSafeContainer& container, const uint64_t guid, const Synapse& synapse,
                   const size_t SpineMaterialId);

    const SynapsesDetails _details;
    Scene& _scene;
};
} // namespace morphology
} // namespace bioexplorer
