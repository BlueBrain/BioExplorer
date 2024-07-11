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

#include "SynapseEfficacySimulationHandler.h"

#include <science/common/Logs.h>

#include <science/io/db/DBConnector.h>

namespace bioexplorer
{
using namespace details;

namespace connectomics
{
using namespace io;
using namespace db;

SynapseEfficacySimulationHandler::SynapseEfficacySimulationHandler(const SynapseEfficacyDetails& details)
    : core::AbstractAnimationHandler()
    , _details(details)
{
    const auto& connector = DBConnector::getInstance();
    _simulationReport = connector.getSimulationReport(_details.populationName, _details.simulationReportId);

    _values = connector.getSynapseEfficacyReportValues(_details.populationName, 0, _details.sqlFilter);

    _frameSize = _values.size();
    _frameData.resize(_frameSize);
    _nbFrames = (_simulationReport.endTime - _simulationReport.startTime) / _simulationReport.timeStep;
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
    PLUGIN_INFO(1, "Start time                  : " << _simulationReport.startTime);
    PLUGIN_INFO(1, "End time                    : " << _simulationReport.endTime);
    PLUGIN_INFO(1, "Time interval               : " << _dt);
    PLUGIN_INFO(1, "Number of frames            : " << _nbFrames);
    PLUGIN_INFO(1, "---------------------------------------------------------");
}

SynapseEfficacySimulationHandler::SynapseEfficacySimulationHandler(const SynapseEfficacySimulationHandler& rhs)
    : core::AbstractAnimationHandler(rhs)
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
        uint64_t i = 0;
        for (const auto& value : _values)
        {
            _frameData[i] = value.second[_currentFrame];
            ++i;
        }
    }

    return _frameData.data();
}

core::AbstractSimulationHandlerPtr SynapseEfficacySimulationHandler::clone() const
{
    return std::make_shared<SynapseEfficacySimulationHandler>(*this);
}
} // namespace connectomics
} // namespace bioexplorer
