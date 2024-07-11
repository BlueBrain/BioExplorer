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

#pragma once

#include <platform/core/common/Types.h>

#include "PointOctreeNode.h"

namespace core
{
/**
 * @brief The PointOctree class implements the PointOctree acceleration structure used by
 * the FieldsRenderer class to render magnetic fields
 *
 */
class PointOctree
{
public:
    /**
     * @brief Construct a new PointOctree object
     *
     * @param points Points used to build the tree. Points contain x, y, z
     * coordinates, as well as a radius, and a value
     * @param voxelSize Voxel size
     * @param minAABB Lower bound of the scene bounding box
     * @param maxAABB Upper bound of the scene bounding box
     */
    PointOctree(const OctreePoints &points, double voxelSize, const Vector3f &minAABB, const Vector3f &maxAABB);

    /**
     * @brief Destroy the PointOctree object
     *
     */
    ~PointOctree();

    /**
     * @brief Get the volume dimentions defined by the scene and the voxel sizes
     *
     * @return The dimensions of the volume
     */
    const Vector3ui &getVolumeDimensions() const { return _volumeDimensions; }

    /**
     * @brief Get the size of the volume
     *
     * @return The size of the volume
     */
    uint32_t getVolumeSize() const { return _volumeSize; }

    /**
     * @brief Get the size of the PointOctree
     *
     * @return The size of the PointOctree
     */
    uint32_t getOctreeSize() const { return _octreeSize; }

    /**
     * @brief Get the depth of the PointOctree
     *
     * @return The depth of the PointOctree
     */
    uint32_t getOctreeDepth() const { return _depth; }

    /**
     * @brief Get a flattened representation of the PointOctree indices
     *
     * @return A flattened representation of the PointOctree indices
     */
    const uint32_ts &getFlatIndices() const { return _flatIndices; }

    /**
     * @brief Get a flattened representation of the PointOctree data (node values)
     *
     * @return A flattened representation of the PointOctree data (node values)
     */
    const floats &getFlatData() const { return _flatData; }

private:
    void _flattenChildren(PointOctreeNode *node, uint32_t level);

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

    Vector3ui _volumeDimensions;
    uint32_t _volumeSize;
    uint32_t _octreeSize;
    uint32_t _depth;

    uint32_ts _offsetPerLevel;

    uint32_ts _flatIndices;
    floats _flatData;
};
} // namespace core
