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

#include "SpikeSimulationHandler.h"

#include <science/common/Logs.h>

#include <science/io/db/DBConnector.h>

namespace bioexplorer
{
namespace morphology
{
using namespace io;
using namespace db;

SpikeSimulationHandler::SpikeSimulationHandler(const std::string& populationName, const uint64_t simulationReportId)
    : core::AbstractAnimationHandler()
    , _populationName(populationName)
    , _simulationReportId(simulationReportId)
{
    const auto& connector = DBConnector::getInstance();
    _simulationReport = connector.getSimulationReport(_populationName, _simulationReportId);
    _nbFrames = (_simulationReport.endTime - _simulationReport.startTime) / _simulationReport.timeStep;
    _dt = _simulationReport.timeStep;

    const auto spikes = connector.getNeuronSpikeReportValues(_populationName, _simulationReportId,
                                                             _simulationReport.startTime, _simulationReport.endTime);

    uint64_t i = 0;
    for (const auto spike : spikes)
    {
        _guidsMapping[spike.first] = i;
        ++i;
    }
    _frameSize = spikes.size();
    _frameData.resize(_frameSize, _restVoltage);

    PLUGIN_INFO(1, "---------------------------------------------------------");
    PLUGIN_INFO(1, "Spike simulation information");
    PLUGIN_INFO(1, "----------------------------");
    PLUGIN_INFO(1, "Population name          : " << _populationName);
    PLUGIN_INFO(1, "Number of simulated nodes: " << _frameSize);
    PLUGIN_INFO(1, "Start time               : " << _simulationReport.startTime);
    PLUGIN_INFO(1, "End time                 : " << _simulationReport.endTime);
    PLUGIN_INFO(1, "Number of frames         : " << _nbFrames);
    _logVisualizationSettings();
    PLUGIN_INFO(1, "---------------------------------------------------------");
}

SpikeSimulationHandler::SpikeSimulationHandler(const SpikeSimulationHandler& rhs)
    : core::AbstractAnimationHandler(rhs)
    , _populationName(rhs._populationName)
    , _simulationReport(rhs._simulationReport)
    , _guidsMapping(rhs._guidsMapping)
{
}

void* SpikeSimulationHandler::getFrameData(const uint32_t frame)
{
    const auto& connector = DBConnector::getInstance();
    const auto boundedFrame = _getBoundedFrame(frame);

    if (_currentFrame != boundedFrame)
    {
        const float ts = _simulationReport.startTime + boundedFrame * _dt;
        const float startTime = ts - (_spikingVoltage - _restVoltage) / _decaySpeed;
        const float endTime = _simulationReport.endTime;
        const auto spikes = connector.getNeuronSpikeReportValues(_populationName, _simulationReportId,
                                                                 std::max(0.f, startTime), std::min(ts, endTime));

        // Rest all values
        for (uint64_t i = 0; i < _frameSize; ++i)
            _frameData[i] = _restVoltage;

        // Update values of nodes spiking within the specified time range
        for (const auto spike : spikes)
            _frameData[_guidsMapping[spike.first]] =
                std::max(_restVoltage, _spikingVoltage - _decaySpeed * (ts - spike.second));

        _currentFrame = boundedFrame;
    }

    return _frameData.data();
}

core::AbstractSimulationHandlerPtr SpikeSimulationHandler::clone() const
{
    return std::make_shared<SpikeSimulationHandler>(*this);
}

void SpikeSimulationHandler::setVisualizationSettings(const double restVoltage, const double spikingVoltage,
                                                      const double decaySpeed)
{
    _restVoltage = restVoltage;
    _spikingVoltage = spikingVoltage;
    _decaySpeed = decaySpeed;
    _logVisualizationSettings();
}

void SpikeSimulationHandler::_logVisualizationSettings()
{
    PLUGIN_INFO(1, "-------------------------");
    PLUGIN_INFO(1, "Rest voltage             : " << _restVoltage);
    PLUGIN_INFO(1, "Spiking voltage          : " << _spikingVoltage);
    PLUGIN_INFO(1, "Decay speed              : " << _decaySpeed);
}

} // namespace morphology
} // namespace bioexplorer
