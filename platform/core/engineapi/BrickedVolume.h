/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Daniel Nachbaur <daniel.nachbaur@epfl.ch>
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

#include <platform/core/common/Api.h>

#include <platform/core/engineapi/Volume.h>

namespace core
{
/**
 * @brief A volume type where the voxels are copied for each added brick.
 * @extends Volume
 */
class BrickedVolume : public virtual Volume
{
public:
    /**
     * @brief Sets a brick of data in the volume.
     * @param data The data to be set as a void pointer.
     * @param position The position of the brick as a Vector3ui object.
     * @param size The size of the brick as a Vector3ui object.
     */
    PLATFORM_API virtual void setBrick(const void* data, const Vector3ui& position, const Vector3ui& size) = 0;

protected:
    /**
     * @brief Constructs a new BrickedVolume object.
     * @param dimensions The dimensions of the volume as a Vector3ui object.
     * @param spacing The spacing between voxels as a Vector3f object.
     * @param type The data type of the volume.
     */
    BrickedVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType type)
        : Volume(dimensions, spacing, type)
    {
    }
};
} // namespace core
