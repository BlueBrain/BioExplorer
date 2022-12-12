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

#include "SomaSimulationHandler.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>

#include <plugin/io/db/DBConnector.h>

namespace bioexplorer
{
namespace morphology
{
using namespace io;
using namespace db;

SomaSimulationHandler::SomaSimulationHandler(const std::string& populationName,
                                             const uint64_t simulationReportId)
    : brayns::AbstractSimulationHandler()
    , _populationName(populationName)
    , _simulationReportId(simulationReportId)
{
    const auto& connector = DBConnector::getInstance();
    _simulationReport =
        connector.getSimulationReport(_populationName, _simulationReportId);

    _frameSize = connector.getNeuronSomaReportNbCells(_populationName,
                                                      _simulationReportId);
    _values =
        connector.getNeuronSomaReportValues(_populationName,
                                            _simulationReportId, _frameSize);
    _frameData.resize(_frameSize);
    _nbFrames = (_simulationReport.endTime - _simulationReport.startTime) /
                _simulationReport.timeStep;
    _dt = _simulationReport.timeStep;
    _logSimulationInformation();
}

void SomaSimulationHandler::_logSimulationInformation()
{
    PLUGIN_INFO(1, "---------------------------------------------------------");
    PLUGIN_INFO(1, "Soma simulation information");
    PLUGIN_INFO(1, "---------------------------");
    PLUGIN_INFO(1, "Population name          : " << _populationName);
    PLUGIN_INFO(1, "Number of simulated nodes: " << _frameSize);
    PLUGIN_INFO(1,
                "Start time               : " << _simulationReport.startTime);
    PLUGIN_INFO(1, "End time                 : " << _simulationReport.endTime);
    PLUGIN_INFO(1, "Time interval            : " << _dt);
    PLUGIN_INFO(1, "Number of frames         : " << _nbFrames);
    PLUGIN_INFO(1, "---------------------------------------------------------");
}

SomaSimulationHandler::SomaSimulationHandler(const SomaSimulationHandler& rhs)
    : brayns::AbstractSimulationHandler(rhs)
    , _populationName(rhs._populationName)
    , _simulationReport(rhs._simulationReport)
    , _guidsMapping(rhs._guidsMapping)
{
}

void* SomaSimulationHandler::getFrameData(const uint32_t frame)
{
    const auto& connector = DBConnector::getInstance();
    const auto boundedFrame = _getBoundedFrame(frame);
    if (_currentFrame != boundedFrame)
    {
        _currentFrame = boundedFrame;
        for (const auto values : _values)
            _frameData[values.first] = values.second[boundedFrame];
    }

    return _frameData.data();
}

brayns::AbstractSimulationHandlerPtr SomaSimulationHandler::clone() const
{
    return std::make_shared<SomaSimulationHandler>(*this);
}
} // namespace morphology
} // namespace bioexplorer
