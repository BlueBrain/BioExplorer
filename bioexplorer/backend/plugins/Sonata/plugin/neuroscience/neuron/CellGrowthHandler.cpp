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

#include "CellGrowthHandler.h"

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
CellGrowthHandler::CellGrowthHandler(const uint32_t nbFrames)
    : core::AbstractAnimationHandler()
{
    // Load simulation information from compartment reports
    _dt = 1.f;
    _nbFrames = nbFrames;
    _unit = "microns";
    _frameSize = nbFrames;
}

CellGrowthHandler::CellGrowthHandler(const CellGrowthHandler& rhs)
    : core::AbstractAnimationHandler(rhs)
{
}

CellGrowthHandler::~CellGrowthHandler() {}

void* CellGrowthHandler::getFrameData(const uint32_t frame)
{
    if (_currentFrame != frame)
    {
        _frameData.resize(_frameSize);
        for (uint64_t i = 0; i < _frameSize; ++i)
            _frameData[i] = (i < frame ? i : _frameSize);
        _currentFrame = frame;
    }
    return _frameData.data();
}

core::AbstractSimulationHandlerPtr CellGrowthHandler::clone() const
{
    return std::make_shared<CellGrowthHandler>(*this);
}
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
