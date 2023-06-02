/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include <common/Logs.h>

#include "SpikeSimulationHandler.h"
#include <platform/core/parameters/AnimationParameters.h>

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
SpikeSimulationHandler::SpikeSimulationHandler(const std::string& reportPath, const brion::GIDSet& gids)
    : AbstractSimulationHandler()
    , _reportPath(reportPath)
    , _gids(gids)
    , _spikeReport(new brain::SpikeReportReader(brain::URI(reportPath), gids))
{
    uint64_t c{0};
    for (const auto gid : _gids)
    {
        _gidMap[gid] = c;
        ++c;
    }

    // Load simulation information from compartment reports
    _nbFrames = _spikeReport->getEndTime() / _timeInterval;
    _dt = _timeInterval;
    _frameSize = _gids.size();
    _frameData.resize(_frameSize, _restVoltage);

    PLUGIN_INFO("-----------------------------------------------------------");
    PLUGIN_INFO("Spike simulation information");
    PLUGIN_INFO("----------------------");
    PLUGIN_INFO("Report path           : " << _reportPath);
    PLUGIN_INFO("Frame size (# of GIDs): " << _frameSize);
    PLUGIN_INFO("End time              : " << _spikeReport->getEndTime());
    PLUGIN_INFO("Time interval         : " << _timeInterval);
    PLUGIN_INFO("Decay speed           : " << _decaySpeed);
    PLUGIN_INFO("Number of frames      : " << _nbFrames);
    PLUGIN_INFO("----------------------");
    PLUGIN_INFO("Report path           : " << _reportPath);
    PLUGIN_INFO("Frame size (# of GIDs): " << _frameSize);
    PLUGIN_INFO("Number of frames      : " << _nbFrames);
    PLUGIN_INFO("End time              : " << _spikeReport->getEndTime());
    _logVisualizationSettings();
    PLUGIN_INFO("-----------------------------------------------------------");
}

SpikeSimulationHandler::SpikeSimulationHandler(const SpikeSimulationHandler& rhs)
    : AbstractSimulationHandler(rhs)
    , _reportPath(rhs._reportPath)
    , _gids(rhs._gids)
    , _spikeReport(rhs._spikeReport)
    , _gidMap(rhs._gidMap)
{
}

void* SpikeSimulationHandler::getFrameData(const uint32_t frame)
{
    const auto boundedFrame = _getBoundedFrame(frame);
    if (_currentFrame != boundedFrame)
    {
        for (size_t i = 0; i < _frameSize; ++i)
        {
            if (_frameData[i] > _restVoltage)
                _frameData[i] -= _decaySpeed;
            else
                _frameData[i] = _restVoltage;
        }

        const float ts = boundedFrame * _dt;
        const float endTime = _spikeReport->getEndTime() - _dt;
        const auto& spikes = _spikeReport->getSpikes(std::min(ts, endTime), std::min(ts + 1.f, endTime));

        for (const auto spike : spikes)
            _frameData[_gidMap[spike.second]] = _spikingVoltage;

        _currentFrame = boundedFrame;
    }

    return _frameData.data();
}

AbstractSimulationHandlerPtr SpikeSimulationHandler::clone() const
{
    return std::make_shared<SpikeSimulationHandler>(*this);
}

void SpikeSimulationHandler::setVisualizationSettings(const double restVoltage, const double spikingVoltage,
                                                      const double timeInterval, const double decaySpeed)
{
    _restVoltage = restVoltage;
    _spikingVoltage = spikingVoltage;
    _timeInterval = timeInterval;
    _decaySpeed = decaySpeed;
    _logVisualizationSettings();
}

void SpikeSimulationHandler::_logVisualizationSettings()
{
    PLUGIN_INFO("----------------------");
    PLUGIN_INFO("Rest voltage          : " << _restVoltage);
    PLUGIN_INFO("Spiking voltage       : " << _spikingVoltage);
    PLUGIN_INFO("Time interval         : " << _timeInterval);
    PLUGIN_INFO("Decay speed           : " << _decaySpeed);
}

} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
