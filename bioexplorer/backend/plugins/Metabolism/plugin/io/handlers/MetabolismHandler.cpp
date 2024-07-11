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

#include "MetabolismHandler.h"

#include <plugin/common/Logs.h>

#include <fstream>

using namespace core;

namespace bioexplorer
{
namespace metabolism
{
MetabolismHandler::MetabolismHandler()
    : core::AbstractAnimationHandler()
{
    // Load simulation information from compartment reports
    _dt = 1.f;
    _unit = "milliseconds";
    _nbFrames = 0;
}

MetabolismHandler::MetabolismHandler(const CommandLineArguments& args)
    : core::AbstractAnimationHandler()
    , _connector(new DBConnector(args))
{
    _dt = 1.f;
    _nbFrames = 0;
    _unit = "ms";
}

MetabolismHandler::MetabolismHandler(const AttachHandlerDetails& payload)
    : core::AbstractAnimationHandler()
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
    : core::AbstractAnimationHandler(rhs)
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
