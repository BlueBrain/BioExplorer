/* Copyright (c) 2018-2021, EPFL/Blue Brain Project
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

#include <plugin/common/Logs.h>

namespace bioexplorer
{
namespace vasculature
{
VasculatureHandler::VasculatureHandler(const VasculatureReportDetails& details,
                                       const uint64_t populationSize)
    : brayns::AbstractSimulationHandler()
    , _details(details)
    , _reader(ElementReportReader(details.path))
    , _report(_reader.openPopulation(details.populationName))
    , _selection({{0, populationSize - 1}})
{
    const auto times = _report.getTimes();
    _startTime = std::get<0>(times);
    const auto endTime = std::get<1>(times);
    _dt = std::get<2>(times);
    _unit = _report.getTimeUnits();

    _nbFrames = (endTime - _startTime) / _dt;
    _frameSize = populationSize;
    _currentFrame = 0;
    PLUGIN_INFO(1, "Report successfully attached");
    PLUGIN_INFO(1, "- Frame size      : " << _frameSize);
    PLUGIN_INFO(1, "- Number of frames: " << _nbFrames);
    PLUGIN_INFO(1, "- Start time      : " << _startTime);
    PLUGIN_INFO(1, "- End time        : " << endTime);
    PLUGIN_INFO(1, "- Time interval   : " << _dt);
    _frameData = _report.get(_selection, _startTime, _startTime + _dt).data;
}

void* VasculatureHandler::getFrameData(const uint32_t frame)
{
    try
    {
        if (_currentFrame != frame && frame < _nbFrames)
        {
            const auto startFrame = _startTime + _dt * frame;
            _frameData = _report.get(_selection, startFrame).data;
        }
    }
    catch (const SonataError& e)
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
