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

#include <plugin/common/Types.h>

#include <brayns/common/simulation/AbstractSimulationHandler.h>

namespace bioexplorer
{
namespace morphology
{
using namespace brayns;
using namespace io;
using namespace common;

/**
 * @brief The SpikeSimulationHandler handles the reading of simulation
 * information from the database at a soma level. When attached to a
 * model, the simulation data is communicated to the renderer by Brayns, and
 * mapped to the geometry by the BioExplorer advanced renderer
 *
 */
class SpikeSimulationHandler : public AbstractSimulationHandler
{
public:
    /** @copydoc AbstractSimulationHandler::AbstractSimulationHandler */
    SpikeSimulationHandler(const std::string& populationName,
                           const uint64_t simulationReportId);

    /** @copydoc AbstractSimulationHandler::AbstractSimulationHandler */
    SpikeSimulationHandler(const SpikeSimulationHandler& rhs);

    /** @copydoc AbstractSimulationHandler::getFrameData */
    void* getFrameData(const uint32_t frame) final;

    /** @copydoc AbstractSimulationHandler::clone */
    brayns::AbstractSimulationHandlerPtr clone() const final;

private:
    std::string _populationName;
    uint64_t _simulationReportId;
    SimulationReport _simulationReport;
    std::map<uint64_t, uint64_t> _guidsMapping;
};
} // namespace morphology
} // namespace bioexplorer
