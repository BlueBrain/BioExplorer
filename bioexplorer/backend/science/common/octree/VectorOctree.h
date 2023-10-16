/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

#pragma once

#include <platform/core/common/Types.h>

#include "VectorOctreeNode.h"

namespace bioexplorer
{
namespace common
{
/**
 * @brief The VectorOctree class implements the VectorOctree acceleration structure used by
 * the FieldsRenderer class to render magnetic fields
 *
 */
class VectorOctree
{
public:
    /**
     * @brief Construct a new VectorOctree object
     *
     * @param events Events used to build the tree. Events contain x, y, z
     * coordinates, as well as a radius, and a value
     * @param voxelSize Voxel size
     * @param minAABB Lower bound of the scene bounding box
     * @param maxAABB Upper bound of the scene bounding box
     */
    VectorOctree(const bioexplorer::common::OctreeVectors &vectors, double voxelSize, const core::Vector3d &minAABB,
                 const core::Vector3d &maxAABB);

    /**
     * @brief Destroy the VectorOctree object
     *
     */
    ~VectorOctree();

    /**
     * @brief Get the volume dimentions defined by the scene and the voxel sizes
     *
     * @return The dimensions of the volume
     */
    const core::Vector3ui &getVolumeDimensions() const { return _volumeDimensions; }

    /**
     * @brief Get the size of the volume
     *
     * @return The size of the volume
     */
    uint32_t getVolumeSize() const { return _volumeSize; }

    /**
     * @brief Get the size of the VectorOctree
     *
     * @return The size of the VectorOctree
     */
    uint32_t getOctreeSize() const { return _octreeSize; }

    /**
     * @brief Get the depth of the VectorOctree
     *
     * @return The depth of the VectorOctree
     */
    uint32_t getOctreeDepth() const { return _depth; }

    /**
     * @brief Get a flattened representation of the VectorOctree indices
     *
     * @return A flattened representation of the VectorOctree indices
     */
    const uint32_ts &getFlatIndices() const { return _flatIndices; }

    /**
     * @brief Get a flattened representation of the VectorOctree data (node values)
     *
     * @return A flattened representation of the VectorOctree data (node values)
     */
    const floats &getFlatData() const { return _flatData; }

private:
    void _flattenChildren(VectorOctreeNode *node, uint32_t level);

    inline uint32_t _pow2roundup(uint32_t x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return x + 1;
    }

    core::Vector3ui _volumeDimensions;
    uint32_t _volumeSize;
    uint32_t _octreeSize;
    uint32_t _depth;

    uint32_ts _offsetPerLevel;

    uint32_ts _flatIndices;
    floats _flatData;
};
} // namespace common
} // namespace bioexplorer