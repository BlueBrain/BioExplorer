/*
 * Copyright (c) 2018, EPFL/Blue Brain Project
 *
 * Responsible Author: Daniel.Nachbaur@epfl.ch
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

#pragma once

#include <platform/core/common/BaseObject.h>
#include <platform/core/common/Types.h>

SERIALIZATION_ACCESS(Statistics)

namespace core
{
/** Captures various statistics about rendering, scenes, etc. */
class Statistics : public BaseObject
{
public:
    double getFPS() const { return _fps; }
    void setFPS(const double fps) { _updateValue(_fps, fps); }
    size_t getSceneSizeInBytes() const { return _sceneSizeInBytes; }
    void setSceneSizeInBytes(const size_t sceneSizeInBytes) { _updateValue(_sceneSizeInBytes, sceneSizeInBytes); }

private:
    double _fps{0.0};
    size_t _sceneSizeInBytes{0};

    SERIALIZATION_FRIEND(Statistics)
};
} // namespace core