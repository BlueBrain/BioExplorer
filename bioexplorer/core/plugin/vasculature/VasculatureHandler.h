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

#include <plugin/api/Params.h>

#include <brayns/api.h>
#include <brayns/common/simulation/AbstractSimulationHandler.h>
#include <brayns/common/types.h>

namespace bioexplorer
{
namespace vasculature
{
/**
 * @brief The VasculatureHandler class handles the mapping of the vasculature
 * simulation to the geometry
 */
class VasculatureHandler : public brayns::AbstractSimulationHandler
{
public:
    /**
     * @brief Default constructor
     */
    VasculatureHandler(const VasculatureReportDetails& details);

    /**
     * @copydoc brayns::AbstractSimulationHandler::getFrameData
     */
    void* getFrameData(const uint32_t) final;

    /**
     * @copydoc brayns::AbstractSimulationHandler::isReady
     */
    bool isReady() const final { return true; }

    /**
     * @copydoc brayns::AbstractSimulationHandler::clone
     */
    brayns::AbstractSimulationHandlerPtr clone() const final;

private:
    VasculatureReportDetails _details;
    std::vector<doubles> _simulationData;
    bool _showVariations{false};

    io::db::SimulationReport _simulationReport;
};
using VasculatureHandlerPtr = std::shared_ptr<VasculatureHandler>;
} // namespace vasculature
} // namespace bioexplorer
