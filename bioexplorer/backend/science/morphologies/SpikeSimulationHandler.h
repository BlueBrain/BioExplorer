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
