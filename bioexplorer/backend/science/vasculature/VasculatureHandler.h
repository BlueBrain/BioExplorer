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

#include <science/api/Params.h>

#include <platform/core/common/Api.h>
#include <platform/core/common/Types.h>
#include <platform/core/common/simulation/AbstractSimulationHandler.h>

namespace bioexplorer
{
namespace vasculature
{
/**
 * @brief The VasculatureHandler class handles the mapping of the vasculature
 * simulation to the geometry
 */
class VasculatureHandler : public core::AbstractSimulationHandler
{
public:
    /**
     * @brief Default constructor
     */
    VasculatureHandler(const details::VasculatureReportDetails& details);

    /**
     * @copydoc core::AbstractSimulationHandler::getFrameData
     */
    void* getFrameData(const uint32_t) final;

    /**
     * @copydoc core::AbstractSimulationHandler::isReady
     */
    bool isReady() const final { return true; }

    /**
     * @copydoc core::AbstractSimulationHandler::clone
     */
    core::AbstractSimulationHandlerPtr clone() const final;

private:
    details::VasculatureReportDetails _details;
    std::vector<doubles> _userData;
    bool _showVariations{false};

    common::SimulationReport _simulationReport;
};
using VasculatureHandlerPtr = std::shared_ptr<VasculatureHandler>;
} // namespace vasculature
} // namespace bioexplorer
