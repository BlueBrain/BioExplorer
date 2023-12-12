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
#include <platform/core/common/BaseObject.h>
#include <platform/core/common/Types.h>

namespace core
{
/**
 * @brief A base class for volumes.
 */
class Volume : public BaseObject
{
public:
    /**
     * @brief Constructs a Volume object.
     * @param dimensions The dimensions of the volume as a Vector3ui object.
     * @param spacing The spacing between voxels as a Vector3f object.
     * @param type The data type of the volume.
     */
    PLATFORM_API Volume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType type);

    /*
     * @brief Sets the range of data in the volume.
     * @param range The data range represented as a Vector2f object.
     */
    PLATFORM_API virtual void setDataRange(const Vector2f& range) = 0;

    /*
     * @brief Gets the range of values in the volume.
     */
    PLATFORM_API Vector2f getDataRange() const { return _valueRange; }

    /**
     * @brief Commits changes to the volume.
     */
    PLATFORM_API virtual void commit() = 0;

    /**
     * @brief Gets the size of the volume in bytes.
     * @return The size of the volume in bytes.
     */
    PLATFORM_API size_t getSizeInBytes() const { return _sizeInBytes; }

    /**
     * @brief Gets the bounding box of the volume.
     * @return The bounding box of the volume as a Boxd object.
     */
    PLATFORM_API Boxd getBounds() const { return {_offset, _offset + Vector3f(_dimensions) * _spacing}; }

    /**
     * @brief Get the Dimensions object
     *
     * @return The dimensions of the volume in the 3D scene
     */
    PLATFORM_API Vector3f getDimensions() const { return _dimensions; }

    /**
     * @brief Get the Element Spacing object
     *
     * @return The voxel size
     */
    PLATFORM_API Vector3f getElementSpacing() const { return _spacing; }

    /**
     * @brief Get the Offset object
     *
     * @return The location of the volume in the 3D scene
     */
    PLATFORM_API Vector3f getOffset() const { return _offset; }

    /**
     * @brief Get the Value Range object
     *
     * @return The range of values in the volume
     */
    PLATFORM_API Vector2f getValueRange() const { return _valueRange; }

protected:
    std::atomic_size_t _sizeInBytes{0}; // The size of the volume in bytes.
    const Vector3ui _dimensions;        // The dimensions of the volume as a Vector3ui object.
    const Vector3f _spacing;            // The spacing between voxels as a Vector3f object.
    Vector3f _offset;                   // Volume offset
    const DataType _dataType;           // The data type of the volume.
    Vector2f _valueRange{-1e6f, 1e6f};  // The voxel value range in the volume.
};
} // namespace core
