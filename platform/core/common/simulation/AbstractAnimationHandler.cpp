/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include "AbstractAnimationHandler.h"

namespace core
{
AbstractAnimationHandler::~AbstractAnimationHandler() = default;

AbstractAnimationHandler& AbstractAnimationHandler::operator=(const AbstractAnimationHandler& rhs)
{
    if (this == &rhs)
        return *this;

    _currentFrame = rhs._currentFrame;
    _nbFrames = rhs._nbFrames;
    _frameSize = rhs._frameSize;
    _dt = rhs._dt;
    _unit = rhs._unit;
    _frameData = rhs._frameData;

    return *this;
}

uint32_t AbstractAnimationHandler::_getBoundedFrame(const uint32_t frame) const
{
    return _nbFrames == 0 ? frame : frame % _nbFrames;
}
} // namespace core
