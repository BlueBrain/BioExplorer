/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
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
