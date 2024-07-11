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

#include "VasculatureHandler.h"

#include <science/io/db/DBConnector.h>

#include <science/common/Logs.h>
#include <science/common/Utils.h>

using namespace core;

namespace bioexplorer
{
using namespace details;
using namespace common;

namespace vasculature
{
using namespace io;
using namespace db;

VasculatureHandler::VasculatureHandler(const VasculatureReportDetails& details)
    : core::AbstractAnimationHandler()
    , _details(details)
{
    auto& connector = DBConnector::getInstance();
    _simulationReport = connector.getSimulationReport(_details.populationName, _details.simulationReportId);
    const auto endTime = _simulationReport.endTime;
    _dt = _simulationReport.timeStep;
    _unit = _simulationReport.timeUnits;
    _nbFrames = (_simulationReport.endTime - _simulationReport.startTime) / _simulationReport.timeStep;
    _frameData = connector.getVasculatureSimulationTimeSeries(_details.populationName, _details.simulationReportId, 0);
    _frameSize = _frameData.size();
    PLUGIN_INFO(1, "Report successfully attached");
    PLUGIN_INFO(1, "- Value evolution : " << boolAsString(_details.showEvolution));
    PLUGIN_INFO(1, "- Frame size      : " << _frameSize);
    PLUGIN_INFO(1, "- Number of frames: " << _nbFrames);
    PLUGIN_INFO(1, "- Start time      : " << _simulationReport.startTime);
    PLUGIN_INFO(1, "- End time        : " << _simulationReport.endTime);
    PLUGIN_INFO(1, "- Time interval   : " << _simulationReport.timeStep);
    PLUGIN_INFO(1, "- Time units      : " << _simulationReport.timeUnits);
}

void* VasculatureHandler::getFrameData(const uint32_t frame)
{
    try
    {
        if (_currentFrame != frame && frame < _nbFrames)
            if (_details.showEvolution)
            {
                const auto startFrame = frame % _nbFrames;
                const auto endFrame = (frame + 1) % _nbFrames;
                const auto startValues =
                    DBConnector::getInstance().getVasculatureSimulationTimeSeries(_details.populationName,
                                                                                  _details.simulationReportId,
                                                                                  startFrame);
                const auto endValues =
                    DBConnector::getInstance().getVasculatureSimulationTimeSeries(_details.populationName,
                                                                                  _details.simulationReportId,
                                                                                  endFrame);

                for (uint64_t i = 0; i < startValues.size(); ++i)
                    _frameData[i] = (endValues[i] - startValues[i]) / startValues[i];
            }
            else
                _frameData =
                    DBConnector::getInstance().getVasculatureSimulationTimeSeries(_details.populationName,
                                                                                  _details.simulationReportId, frame);
    }
    catch (const std::runtime_error& e)
    {
        PLUGIN_ERROR(e.what())
    }
    _currentFrame = frame;
    return _frameData.data();
}

core::AbstractSimulationHandlerPtr VasculatureHandler::clone() const
{
    return std::make_shared<VasculatureHandler>(*this);
}
} // namespace vasculature
} // namespace bioexplorer
