/*
 * Copyright 2020-2023 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "MetabolismHandler.h"

#include <plugin/common/Logs.h>

#include <fstream>

using namespace core;

namespace bioexplorer
{
namespace metabolism
{
MetabolismHandler::MetabolismHandler()
    : core::AbstractSimulationHandler()
{
    // Load simulation information from compartment reports
    _dt = 1.f;
    _unit = "milliseconds";
    _nbFrames = 0;
}

MetabolismHandler::MetabolismHandler(const CommandLineArguments& args)
    : core::AbstractSimulationHandler()
    , _connector(new DBConnector(args))
{
    _dt = 1.f;
    _nbFrames = 0;
    _unit = "ms";
}

MetabolismHandler::MetabolismHandler(const AttachHandlerDetails& payload)
    : core::AbstractSimulationHandler()
    , _connector(new DBConnector(payload))
{
    _dt = 1.f;
    _nbFrames = _connector->getNbFrames();
    _locations = _connector->getLocations();
    _unit = "ms";
    _metaboliteIds = payload.metaboliteIds;
    _relativeConcentration = payload.relativeConcentration;
    _referenceFrame = payload.referenceFrame;
    std::string metabolitesIds;
    for (const auto metaboliteId : _metaboliteIds)
    {
        if (!metabolitesIds.empty())
            metabolitesIds += ",";
        metabolitesIds += std::to_string(metaboliteId);
    }
    PLUGIN_INFO("Setting metabolites: " << metabolitesIds);
}

MetabolismHandler::MetabolismHandler(const MetabolismHandler& rhs)
    : core::AbstractSimulationHandler(rhs)
{
}

MetabolismHandler::~MetabolismHandler() {}

void* MetabolismHandler::getFrameData(const uint32_t frame)
{
    if (_currentFrame == frame)
        return _frameData.data();
    _currentFrame = frame;

    const auto values = _connector->getConcentrations(frame, _referenceFrame, _metaboliteIds, _relativeConcentration);

    _frameData.clear();
    _frameData.push_back(frame);
    size_t j = 0;
    for (size_t i = 0; i < _locations.size(); ++i)
    {
        const auto idx = values.find(_locations[i].guid);
        if (idx != values.end())
        {
            _frameData.push_back((*idx).second);
            ++j;
        }
        else
            _frameData.push_back(1e38f);
    }

    _frameSize = _frameData.size();
    return _frameData.data();
}

AbstractSimulationHandlerPtr MetabolismHandler::clone() const
{
    return std::make_shared<MetabolismHandler>(*this);
}
} // namespace metabolism
} // namespace bioexplorer
