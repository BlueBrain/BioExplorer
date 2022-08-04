/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include "CompartmentSimulationHandler.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>

#include <plugin/io/db/DBConnector.h>

namespace
{
const float DEFAULT_REST_VALUE = 0.f;
const float DEFAULT_SPIKING_VALUE = 1.f;
const float DEFAULT_TIME_INTERVAL = 0.01f;
const float DEFAULT_DECAY_SPEED = 0.01f;
} // namespace

namespace bioexplorer
{
namespace morphology
{
using namespace io;
using namespace db;

CompartmentSimulationHandler::CompartmentSimulationHandler(
    const std::string& populationName, const uint64_t simulationReportId)
    : brayns::AbstractSimulationHandler()
    , _populationName(populationName)
    , _simulationReportId(simulationReportId)
{
    PLUGIN_THROW("Not implemented");
}

CompartmentSimulationHandler::CompartmentSimulationHandler(
    const CompartmentSimulationHandler& rhs)
    : brayns::AbstractSimulationHandler(rhs)
    , _populationName(rhs._populationName)
    , _simulationReport(rhs._simulationReport)
{
}

void* CompartmentSimulationHandler::getFrameData(const uint32_t frame)
{
    PLUGIN_THROW("Not implemented");
    return _frameData.data();
}

brayns::AbstractSimulationHandlerPtr CompartmentSimulationHandler::clone() const
{
    return std::make_shared<CompartmentSimulationHandler>(*this);
}
} // namespace morphology
} // namespace bioexplorer
