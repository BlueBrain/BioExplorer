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

#include <science/common/Types.h>

#include <platform/core/common/simulation/AbstractSimulationHandler.h>

namespace bioexplorer
{
namespace connectomics
{
using namespace core;
using namespace common;
using namespace details;

/**
 * @brief The SynapseEfficacySimulationHandler handles the reading of simulation
 * information from the database at a soma level. When attached to a
 * model, the simulation data is communicated to the renderer by Core, and
 * mapped to the geometry by the BioExplorer advanced renderer
 *
 */
class SynapseEfficacySimulationHandler : public AbstractSimulationHandler
{
public:
    /** @copydoc AbstractSimulationHandler::AbstractSimulationHandler */
    SynapseEfficacySimulationHandler(const SynapseEfficacyDetails& details);
    /** @copydoc AbstractSimulationHandler::AbstractSimulationHandler */
    SynapseEfficacySimulationHandler(const SynapseEfficacySimulationHandler& rhs);

    /** @copydoc AbstractSimulationHandler::getFrameData */
    void* getFrameData(const uint32_t frame) final;

    /** @copydoc AbstractSimulationHandler::clone */
    core::AbstractSimulationHandlerPtr clone() const final;

private:
    void _logSimulationInformation();

    SynapseEfficacyDetails _details;
    SimulationReport _simulationReport;

    std::map<uint64_t, floats> _values;
};
} // namespace connectomics
} // namespace bioexplorer
