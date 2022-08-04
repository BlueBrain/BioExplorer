/* Copyright (c) 2020-2022, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "VasculatureHandler.h"

#include <plugin/io/db/DBConnector.h>

#include <plugin/common/Logs.h>

namespace bioexplorer
{
namespace vasculature
{
using namespace io;
using namespace db;

VasculatureHandler::VasculatureHandler(const VasculatureReportDetails& details)
    : brayns::AbstractSimulationHandler()
    , _details(details)
{
    auto& connector = DBConnector::getInstance();
    _simulationReport =
        connector.getSimulationReport(_details.populationName,
                                      _details.simulationReportId);
    const auto endTime = _simulationReport.endTime;
    _dt = _simulationReport.timeStep;
    _unit = _simulationReport.timeUnits;
    _nbFrames = (_simulationReport.endTime - _simulationReport.startTime) /
                _simulationReport.timeStep;
    _frameData = connector.getVasculatureSimulationTimeSeries(
        _details.populationName, _details.simulationReportId, 0);
    _frameSize = _frameData.size();
    PLUGIN_INFO(1, "Report successfully attached");
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
            _frameData =
                DBConnector::getInstance().getVasculatureSimulationTimeSeries(
                    _details.populationName, _details.simulationReportId,
                    frame);
    }
    catch (const std::runtime_error& e)
    {
        PLUGIN_ERROR(e.what())
    }
    _currentFrame = frame;
    return _frameData.data();
}

brayns::AbstractSimulationHandlerPtr VasculatureHandler::clone() const
{
    return std::make_shared<VasculatureHandler>(*this);
}
} // namespace vasculature
} // namespace bioexplorer
