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
 * @brief A volume type where the voxels are computed in real-time using a pre-loaded Octree structure of events (3D
 * position and value)
 * @extends Volume
 */
class OctreeVolume : public virtual Volume
{
public:
    /**
     * @brief Set the Octree object
     *
     * @param offset Location of the octree in the 3D scene
     * @param indices Indices of the Octree
     * @param values Values of the Octree
     * @param dataType Data type
     */
    PLATFORM_API virtual void setOctree(const Vector3f& offset, const uint32_ts& indices, const floats& values,
                                        const OctreeDataType dataType) = 0;

    /**
     * @brief Get the Octree Indices object
     *
     * @return const uint32_ts& Indices in the octree
     */
    PLATFORM_API const uint32_ts& getOctreeIndices() const { return _octreeIndices; }

    /**
     * @brief Get the Octree Values object
     *
     * @return const floats& Values in the octree
     */
    PLATFORM_API const floats& getOctreeValues() const { return _octreeValues; }

    /**
     * @brief Get the Octree Data Type object
     *
     * @return OctreeDataType The data type
     */
    PLATFORM_API OctreeDataType getOctreeDataType() const { return _octreeDataType; }

protected:
    /**
     * @brief Constructs a new BrickedVolume object.
     * @param dimensions The dimensions of the volume as a Vector3ui object.
     * @param spacing The spacing between voxels as a Vector3f object.
     * @param type The data type of the volume.
     */
    OctreeVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType type)
        : Volume(dimensions, spacing, type)
    {
    }

    // Octree
    uint32_ts _octreeIndices;
    floats _octreeValues;
    OctreeDataType _octreeDataType;
};
} // namespace core
