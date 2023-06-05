/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "SomaSimulationHandler.h"

#include <science/common/Logs.h>

#include <science/io/db/DBConnector.h>

namespace bioexplorer
{
namespace morphology
{
using namespace io;
using namespace db;

SomaSimulationHandler::SomaSimulationHandler(const std::string& populationName, const uint64_t simulationReportId)
    : core::AbstractSimulationHandler()
    , _populationName(populationName)
    , _simulationReportId(simulationReportId)
{
    const auto& connector = DBConnector::getInstance();
    _simulationReport = connector.getSimulationReport(_populationName, _simulationReportId);

    _nbFrames = (_simulationReport.endTime - _simulationReport.startTime) / _simulationReport.timeStep;
    _dt = _simulationReport.timeStep;
    _logSimulationInformation();
}

void SomaSimulationHandler::_logSimulationInformation()
{
    PLUGIN_INFO(1, "---------------------------------------------------------");
    PLUGIN_INFO(1, "Soma simulation information");
    PLUGIN_INFO(1, "---------------------------");
    PLUGIN_INFO(1, "Population name          : " << _populationName);
    PLUGIN_INFO(1, "Start time               : " << _simulationReport.startTime);
    PLUGIN_INFO(1, "End time                 : " << _simulationReport.endTime);
    PLUGIN_INFO(1, "Time interval            : " << _dt);
    PLUGIN_INFO(1, "Number of frames         : " << _nbFrames);
    PLUGIN_INFO(1, "---------------------------------------------------------");
}

SomaSimulationHandler::SomaSimulationHandler(const SomaSimulationHandler& rhs)
    : core::AbstractSimulationHandler(rhs)
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
        if (_simulationReport.debugMode)
        {
            for (uint64_t i = 0; i < _frameData.size(); ++i)
                _frameData[i] = -70.f + 100 * cos((frame + i) * M_PI / 180.f);
        }
        else
            connector.getNeuronSomaReportValues(_populationName, _simulationReportId, _currentFrame, _frameData);
        _frameSize = _frameData.size();
    }

    return _frameData.data();
}

core::AbstractSimulationHandlerPtr SomaSimulationHandler::clone() const
{
    return std::make_shared<SomaSimulationHandler>(*this);
}
} // namespace morphology
} // namespace bioexplorer
