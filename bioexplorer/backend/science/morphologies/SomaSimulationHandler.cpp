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
    : core::AbstractAnimationHandler()
    , _populationName(populationName)
    , _simulationReportId(simulationReportId)
{
    const auto& connector = DBConnector::getInstance();
    _simulationReport = connector.getSimulationReport(_populationName, _simulationReportId);
    _frameSize = _simulationReport.guids.size();
    if (_simulationReport.guids.empty())
        _frameSize = connector.getNumberOfNeurons(populationName);

    _nbFrames = (_simulationReport.endTime - _simulationReport.startTime) / _simulationReport.timeStep;
    _frameData.resize(_frameSize);
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
    : core::AbstractAnimationHandler(rhs)
    , _populationName(rhs._populationName)
    , _simulationReport(rhs._simulationReport)
    , _guidsMapping(rhs._guidsMapping)
{
}

void* SomaSimulationHandler::getFrameData(const uint32_t frame)
{
    const auto boundedFrame = _getBoundedFrame(frame);
    if (_currentFrame != boundedFrame)
    {
        _currentFrame = boundedFrame;
        if (_simulationReport.debugMode)
        {
            for (uint64_t i = 0; i < _frameSize; ++i)
            {
                const float voltage = -100.f + 100.f * cos((frame + i) * M_PI / 180.f);
                _frameData[i] = std::max(-80.f, voltage);
            }
        }
        else
        {
            const auto& connector = DBConnector::getInstance();
            connector.getNeuronSomaReportValues(_populationName, _simulationReportId, _currentFrame, _frameData);
            _frameSize = _frameData.size();
        }
    }

    return _frameData.data();
}

core::AbstractSimulationHandlerPtr SomaSimulationHandler::clone() const
{
    return std::make_shared<SomaSimulationHandler>(*this);
}
} // namespace morphology
} // namespace bioexplorer
