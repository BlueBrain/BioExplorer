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

#include <science/common/Types.h>

#include <platform/core/common/simulation/AbstractAnimationHandler.h>

namespace bioexplorer
{
namespace connectomics
{
/**
 * @brief The SynapseEfficacySimulationHandler handles the reading of simulation
 * information from the database at a soma level. When attached to a
 * model, the simulation data is communicated to the renderer by Core, and
 * mapped to the geometry by the BioExplorer advanced renderer
 *
 */
class SynapseEfficacySimulationHandler : public core::AbstractAnimationHandler
{
public:
    /** @copydoc AbstractAnimationHandler::AbstractAnimationHandler */
    SynapseEfficacySimulationHandler(const details::SynapseEfficacyDetails& details);

    /** @copydoc AbstractAnimationHandler::AbstractAnimationHandler */
    SynapseEfficacySimulationHandler(const SynapseEfficacySimulationHandler& rhs);

    /** @copydoc AbstractAnimationHandler::getFrameData */
    void* getFrameData(const uint32_t frame) final;

    /** @copydoc AbstractAnimationHandler::clone */
    core::AbstractSimulationHandlerPtr clone() const final;

private:
    void _logSimulationInformation();

    details::SynapseEfficacyDetails _details;
    common::SimulationReport _simulationReport;

    std::map<uint64_t, floats> _values;
};
} // namespace connectomics
} // namespace bioexplorer
