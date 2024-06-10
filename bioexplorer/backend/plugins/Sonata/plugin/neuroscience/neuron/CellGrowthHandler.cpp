/*
 * Copyright 2020-2024 Blue Brain Project / EPFL
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
