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

#include "SynapseEfficacySimulationHandler.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>

#include <plugin/io/db/DBConnector.h>

namespace bioexplorer
{
namespace connectomics
{
using namespace io;
using namespace db;

SynapseEfficacySimulationHandler::SynapseEfficacySimulationHandler(
    const SynapseEfficacyDetails& details)
    : brayns::AbstractSimulationHandler()
    , _details(details)
{
    const auto& connector = DBConnector::getInstance();
    _simulationReport =
        connector.getSimulationReport(_details.populationName,
                                      _details.simulationReportId);

    const auto synapsePositions =
        DBConnector::getInstance().getSynapseEfficacyPositions(
            _details.populationName, _details.sqlFilter);

    _frameSize = synapsePositions.size();
    _frameData.resize(_frameSize);
    _nbFrames = (_simulationReport.endTime - _simulationReport.startTime) /
                _simulationReport.timeStep;
    _dt = _simulationReport.timeStep;
    _logSimulationInformation();
}

void SynapseEfficacySimulationHandler::_logSimulationInformation()
{
    PLUGIN_INFO(1, "---------------------------------------------------------");
    PLUGIN_INFO(1, "Synapse efficacy simulation information");
    PLUGIN_INFO(1, "---------------------------------------");
    PLUGIN_INFO(1, "Population name             : " << _details.populationName);
    PLUGIN_INFO(1, "Number of simulated synapses: " << _frameSize);
    PLUGIN_INFO(1, "Start time                  : "
                       << _simulationReport.startTime);
    PLUGIN_INFO(1,
                "End time                    : " << _simulationReport.endTime);
    PLUGIN_INFO(1, "Time interval               : " << _dt);
    PLUGIN_INFO(1, "Number of frames            : " << _nbFrames);
    PLUGIN_INFO(1, "---------------------------------------------------------");
}

SynapseEfficacySimulationHandler::SynapseEfficacySimulationHandler(
    const SynapseEfficacySimulationHandler& rhs)
    : brayns::AbstractSimulationHandler(rhs)
    , _details(rhs._details)
{
}

void* SynapseEfficacySimulationHandler::getFrameData(const uint32_t frame)
{
    const auto& connector = DBConnector::getInstance();
    const auto boundedFrame = _getBoundedFrame(frame);
    if (_currentFrame != boundedFrame)
    {
        _currentFrame = boundedFrame;
        const auto values =
            connector.getSynapseEfficacyReportValues(_details.populationName,
                                                     _currentFrame,
                                                     _details.sqlFilter);

        if( values.size() != _frameData.size())
            _frameData.resize(values.size());
            
        uint64_t i = 0;
        for (const auto& value : values)
        {
            _frameData[i] = value;
            ++i;
        }

        _frameSize = _frameData.size();
    }

    return _frameData.data();
}

brayns::AbstractSimulationHandlerPtr SynapseEfficacySimulationHandler::clone()
    const
{
    return std::make_shared<SynapseEfficacySimulationHandler>(*this);
}
} // namespace connectomics
} // namespace bioexplorer
