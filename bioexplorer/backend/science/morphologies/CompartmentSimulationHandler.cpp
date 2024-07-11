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

#include "CompartmentSimulationHandler.h"

#include <science/common/Logs.h>

#include <science/io/db/DBConnector.h>

namespace
{
const float DEFAULT_REST_VALUE = 0.f;
const float DEFAULT_SPIKING_VALUE = 1.f;
const float DEFAULT_TIME_INTERVAL = 0.01f;
const float DEFAULT_DECAY_SPEED = 0.01f;
} // namespace

namespace bioexplorer
{
namespace morphology
{
using namespace io;
using namespace db;

CompartmentSimulationHandler::CompartmentSimulationHandler(const std::string& populationName,
                                                           const uint64_t simulationReportId)
    : core::AbstractAnimationHandler()
    , _populationName(populationName)
    , _simulationReportId(simulationReportId)
{
    const auto& connector = DBConnector::getInstance();
    _simulationReport = connector.getSimulationReport(_populationName, _simulationReportId);
    _nbFrames = (_simulationReport.endTime - _simulationReport.startTime) / _simulationReport.timeStep;
    _dt = _simulationReport.timeStep;

    const auto values = connector.getNeuronCompartmentReportValues(_populationName, _simulationReportId, 0);

    _frameSize = values.size();
    _frameData.resize(_frameSize);
    mempcpy(&_frameData.data()[0], values.data(), sizeof(float) * values.size());

    PLUGIN_INFO(1, "---------------------------------------------------------");
    PLUGIN_INFO(1, "Compartment simulation information");
    PLUGIN_INFO(1, "----------------------------------");
    PLUGIN_INFO(1, "Population name          : " << _populationName);
    PLUGIN_INFO(1, "Number of simulated nodes: " << _frameSize);
    PLUGIN_INFO(1, "Start time               : " << _simulationReport.startTime);
    PLUGIN_INFO(1, "End time                 : " << _simulationReport.endTime);
    PLUGIN_INFO(1, "Time interval            : " << _dt);
    PLUGIN_INFO(1, "Number of frames         : " << _nbFrames);
    PLUGIN_INFO(1, "---------------------------------------------------------");
}

CompartmentSimulationHandler::CompartmentSimulationHandler(const CompartmentSimulationHandler& rhs)
    : core::AbstractAnimationHandler(rhs)
    , _populationName(rhs._populationName)
    , _simulationReport(rhs._simulationReport)
{
}

void* CompartmentSimulationHandler::getFrameData(const uint32_t frame)
{
    const auto& connector = DBConnector::getInstance();
    const auto boundedFrame = _getBoundedFrame(frame);
    if (_currentFrame != boundedFrame)
    {
        const auto values = connector.getNeuronCompartmentReportValues(_populationName, _simulationReportId, frame);
        mempcpy(&_frameData.data()[0], values.data(), sizeof(float) * values.size());
        _currentFrame = boundedFrame;
    }

    return _frameData.data();
}

core::AbstractSimulationHandlerPtr CompartmentSimulationHandler::clone() const
{
    return std::make_shared<CompartmentSimulationHandler>(*this);
}
} // namespace morphology
} // namespace bioexplorer
