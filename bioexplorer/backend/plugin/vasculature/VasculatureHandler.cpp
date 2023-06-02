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

#include "VasculatureHandler.h"

#include <plugin/io/db/DBConnector.h>

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

namespace bioexplorer
{
namespace vasculature
{
using namespace io;
using namespace db;

VasculatureHandler::VasculatureHandler(const VasculatureReportDetails& details)
    : core::AbstractSimulationHandler()
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
