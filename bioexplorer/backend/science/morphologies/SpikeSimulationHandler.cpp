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
    : core::AbstractSimulationHandler()
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
    : core::AbstractSimulationHandler(rhs)
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
