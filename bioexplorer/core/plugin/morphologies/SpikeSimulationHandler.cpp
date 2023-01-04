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

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Logs.h>

#include <plugin/io/db/DBConnector.h>

namespace
{
const float DEFAULT_REST_VALUE = -65.f;
const float DEFAULT_SPIKING_VALUE = -10.f;
const float DEFAULT_DECAY_SPEED = 5.0f;
const float DEFAULT_TIME_INTERVAL = 0.01f;
} // namespace

namespace bioexplorer
{
namespace morphology
{
using namespace io;
using namespace db;

SpikeSimulationHandler::SpikeSimulationHandler(
    const std::string& populationName, const uint64_t simulationReportId)
    : brayns::AbstractSimulationHandler()
    , _populationName(populationName)
    , _simulationReportId(simulationReportId)
{
    const auto& connector = DBConnector::getInstance();
    _simulationReport =
        connector.getSimulationReport(_populationName, _simulationReportId);
    _nbFrames = (_simulationReport.endTime - _simulationReport.startTime) /
                _simulationReport.timeStep;
    _dt = _simulationReport.timeStep;

    const auto spikes = connector.getNeuronSpikeReportValues(
        _populationName, _simulationReportId, _simulationReport.startTime,
        _simulationReport.endTime + _simulationReport.timeStep);

    uint64_t i = 0;
    for (const auto spike : spikes)
    {
        _guidsMapping[spike] = i;
        ++i;
    }
    _frameSize = spikes.size();
    _frameData.resize(_frameSize, DEFAULT_REST_VALUE);

    PLUGIN_INFO(1, "---------------------------------------------------------");
    PLUGIN_INFO(1, "Spike simulation information");
    PLUGIN_INFO(1, "----------------------------");
    PLUGIN_INFO(1, "Population name          : " << _populationName);
    PLUGIN_INFO(1, "Number of simulated nodes: " << _frameSize);
    PLUGIN_INFO(1,
                "Start time               : " << _simulationReport.startTime);
    PLUGIN_INFO(1, "End time                 : " << _simulationReport.endTime);
    PLUGIN_INFO(1, "Time interval            : " << _simulationReport.timeStep);
    PLUGIN_INFO(1, "Decay speed              : " << DEFAULT_DECAY_SPEED);
    PLUGIN_INFO(1, "Number of frames         : " << _nbFrames);
    PLUGIN_INFO(1, "---------------------------------------------------------");
}

SpikeSimulationHandler::SpikeSimulationHandler(
    const SpikeSimulationHandler& rhs)
    : brayns::AbstractSimulationHandler(rhs)
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
        for (size_t i = 0; i < _frameSize; ++i)
        {
            if (_frameData[i] > DEFAULT_REST_VALUE)
                _frameData[i] -= DEFAULT_DECAY_SPEED;
            else
                _frameData[i] = DEFAULT_REST_VALUE;
        }

        const double ts = _simulationReport.startTime + boundedFrame * _dt;
        const double endTime = _simulationReport.endTime - _dt;
        const auto spikes =
            connector.getNeuronSpikeReportValues(_populationName,
                                                 _simulationReportId,
                                                 std::min(ts, endTime),
                                                 std::min(ts + _dt, endTime));

        for (const auto spike : spikes)
            _frameData[_guidsMapping[spike]] = DEFAULT_SPIKING_VALUE;

        _currentFrame = boundedFrame;
    }

    return _frameData.data();
}

brayns::AbstractSimulationHandlerPtr SpikeSimulationHandler::clone() const
{
    return std::make_shared<SpikeSimulationHandler>(*this);
}
} // namespace morphology
} // namespace bioexplorer
