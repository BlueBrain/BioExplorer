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

#include <common/Logs.h>

#include "SpikeSimulationHandler.h"
#include <platform/core/parameters/AnimationParameters.h>

using namespace core;

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
SpikeSimulationHandler::SpikeSimulationHandler(const std::string& reportPath, const brion::GIDSet& gids)
    : AbstractAnimationHandler()
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
    : AbstractAnimationHandler(rhs)
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
