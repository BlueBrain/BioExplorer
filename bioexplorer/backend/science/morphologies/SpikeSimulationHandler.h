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
namespace morphology
{
/**
 * @brief The SpikeSimulationHandler handles the reading of simulation
 * information from the database at a soma level. When attached to a
 * model, the simulation data is communicated to the renderer by Core, and
 * mapped to the geometry by the BioExplorer advanced renderer
 *
 */
class SpikeSimulationHandler : public core::AbstractAnimationHandler
{
public:
    /** @copydoc AbstractAnimationHandler::AbstractAnimationHandler */
    SpikeSimulationHandler(const std::string& populationName, const uint64_t simulationReportId);

    /** @copydoc AbstractAnimationHandler::AbstractAnimationHandler */
    SpikeSimulationHandler(const SpikeSimulationHandler& rhs);

    /** @copydoc AbstractAnimationHandler::getFrameData */
    void* getFrameData(const uint32_t frame) final;

    /** @copydoc AbstractAnimationHandler::clone */
    core::AbstractSimulationHandlerPtr clone() const final;

    /** Sets the visualization settings */
    void setVisualizationSettings(const double restVoltage, const double spikingVoltage, const double decaySpeed);

private:
    void _logVisualizationSettings();

    std::string _populationName;
    uint64_t _simulationReportId;
    common::SimulationReport _simulationReport;

    float _restVoltage{-65.f};
    float _spikingVoltage{-10.f};
    float _decaySpeed{1.f};

    std::map<uint64_t, uint64_t> _guidsMapping;
};
} // namespace morphology
} // namespace bioexplorer
