/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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
