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

#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>

#include <limits>

namespace bioexplorer
{
namespace vasculature
{
VasculatureHandler::VasculatureHandler(const VasculatureReportDetails& details)
    : brayns::AbstractSimulationHandler()
    , _details(details)
{
    // Load simulation information from compartment reports
    _dt = 1.f;

    if (_details.debug)
    {
        _frameSize = 1349411;
        _nbFrames = 720;
        _unit = "ms (debug)";
    }
    else
    {
        _nbFrames = 1;
        _frameSize = 0;
        _unit = "ms";

        try
        {
            std::unique_ptr<HighFive::File> file =
                std::unique_ptr<HighFive::File>(
                    new HighFive::File(_details.path,
                                       HighFive::File::ReadOnly));
            const auto& report = file->getGroup("report");
            const auto& vasculature = report.getGroup("vasculature");
            const auto& dataset = vasculature.getDataSet("data");
            dataset.read(_simulationData);
            _nbFrames = _simulationData.size();
            if (_nbFrames == 0)
                PLUGIN_THROW("Report file does not contain any data: " +
                             _details.path);

            _frameSize = _simulationData[0].size();
            float minValue = std::numeric_limits<float>::max();
            float maxValue = -std::numeric_limits<float>::max();
            for (const auto& series : _simulationData)
                for (const auto& value : series)
                {
                    const float v = static_cast<float>(value);
                    minValue = std::min(minValue, v);
                    maxValue = std::max(maxValue, v);
                }
            PLUGIN_INFO(1, "Report successfully attached. Frame size is "
                               << _frameSize << ". Values are in range ["
                               << minValue << "," << maxValue << "]");
        }
        catch (const HighFive::FileException& exc)
        {
            PLUGIN_THROW("Could not open vasculature report file " +
                         _details.path + ": " + exc.what());
        }
    }
}

VasculatureHandler::VasculatureHandler(const VasculatureHandler& rhs)
    : brayns::AbstractSimulationHandler(rhs)
{
}

VasculatureHandler::~VasculatureHandler() {}

void* VasculatureHandler::getFrameData(const uint32_t frame)
{
    if (_currentFrame != frame)
    {
        _frameData.clear();
        _frameData.reserve(_frameSize);
        if (_details.debug)
        {
            for (uint64_t i = 0; i < _frameSize; ++i)
            {
                const float value =
                    0.5f +
                    0.5f * (sin(float(frame + i) * M_PI / 360.f) +
                            0.5f * cos(float(frame + i) * 3.f * M_PI / 360.f));
                _frameData.push_back(static_cast<float>(value));
            }
        }
        else
        {
            const auto& frameData = _simulationData[frame];
            for (const auto value : frameData)
                _frameData.push_back(static_cast<float>(value));
        }
        _currentFrame = frame;
        PLUGIN_DEBUG("Frame " << frame << " loaded: " << _frameData.size()
                              << " segments");
    }
    return _frameData.data();
}

brayns::AbstractSimulationHandlerPtr VasculatureHandler::clone() const
{
    return std::make_shared<VasculatureHandler>(*this);
}
} // namespace vasculature
} // namespace bioexplorer
