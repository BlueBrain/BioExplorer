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

#include "VectorOctree.h"

#include <platform/core/common/CommonTypes.h>
#include <platform/core/common/Logs.h>

namespace core
{
typedef std::map<uint32_t, VectorOctreeNode> VectorOctreeLevelMap;

VectorOctree::VectorOctree(const OctreeVectors &vectors, double voxelSize, const Vector3d &minAABB,
                           const Vector3d &maxAABB)
    : _volumeDimensions(Vector3ui(0u, 0u, 0u))
    , _volumeSize(0u)
{
    CORE_INFO("Nb of vectors : " << vectors.size());

    // **************** VectorOctree creations *******************
    // *****************************************************
    Vector3ui octreeSize(_pow2roundup(std::ceil((maxAABB.x - minAABB.x) / voxelSize)),
                         _pow2roundup(std::ceil((maxAABB.y - minAABB.y) / voxelSize)),
                         _pow2roundup(std::ceil((maxAABB.z - minAABB.z) / voxelSize)));

    // This octree is always cubic
    _octreeSize = std::max(std::max(octreeSize.x, octreeSize.y), octreeSize.z);

    CORE_INFO("Vector Octree size  : " << _octreeSize);

    _depth = std::log2(_octreeSize) + 1u;
    std::vector<VectorOctreeLevelMap> octree(_depth);

    CORE_INFO("Vector Octree depth : " << _depth << " " << octree.size());

    if (_depth == 0)
        return;

    for (uint32_t i = 0; i < vectors.size(); ++i)
    {
        CORE_PROGRESS("Bulding Vector Octree from vectors", i, vectors.size());
        const uint32_t xpos = std::floor((vectors[i].position.x - minAABB.x) / voxelSize);
        const uint32_t ypos = std::floor((vectors[i].position.y - minAABB.y) / voxelSize);
        const uint32_t zpos = std::floor((vectors[i].position.z - minAABB.z) / voxelSize);
        const Vector3d &value = vectors[i].direction;

        const uint32_t indexX = xpos;
        const uint32_t indexY = ypos * (uint32_t)_octreeSize;
        const uint32_t indexZ = zpos * (uint32_t)_octreeSize * (uint32_t)_octreeSize;

        auto it = octree[0].find(indexX + indexY + indexZ);
        if (it == octree[0].end())
        {
            VectorOctreeNode *child = nullptr;
            for (uint32_t level = 0; level < _depth; ++level)
            {
                bool newNode = false;
                const uint32_t divisor = std::pow(2, level);
                const Vector3f center(xpos, ypos, zpos);

                const uint32_t nBlock = _octreeSize / divisor;
                const uint32_t index = std::floor(xpos / divisor) + nBlock * std::floor(ypos / divisor) +
                                       nBlock * nBlock * std::floor(zpos / divisor);

                const double size = voxelSize * (level + 1u);

                if (octree[level].find(index) == octree[level].end())
                {
                    octree[level].insert(VectorOctreeLevelMap::value_type(index, VectorOctreeNode(center, size)));
                    newNode = true;
                }

                octree[level].at(index).addValue(value);

                if ((level > 0) && (child != nullptr))
                    octree[level].at(index).setChild(child);

                if (newNode)
                    child = &(octree[level].at(index));
                else
                    child = nullptr;
            }
        }
        else
        {
            for (uint32_t level = 0; level < _depth; ++level)
            {
                const uint32_t divisor = std::pow(2, level);
                const uint32_t nBlock = _octreeSize / divisor;
                const uint32_t index = std::floor(xpos / divisor) + nBlock * std::floor(ypos / divisor) +
                                       nBlock * nBlock * std::floor(zpos / divisor);
                octree[level].at(index).addValue(value);
            }
        }
    }
    for (uint32_t i = 0; i < octree.size(); ++i)
        CORE_DEBUG("Number of leaves [" << i << "]: " << octree[i].size());

    // VectorOctree flattening
    _offsetPerLevel.resize(_depth);
    _offsetPerLevel[_depth - 1u] = 0;
    uint32_t previousOffset = 0u;
    for (uint32_t i = _depth - 1u; i > 0u; --i)
    {
        _offsetPerLevel[i - 1u] = previousOffset + octree[i].size();
        previousOffset = _offsetPerLevel[i - 1u];
    }

    uint32_t totalNodeNumber = 0;

    for (uint32_t i = 0; i < octree.size(); ++i)
        totalNodeNumber += octree[i].size();

    // Needs to be initialized with zeros
    _flatIndices.resize(totalNodeNumber * 2u, 0);
    _flatData.resize(totalNodeNumber * FIELD_VECTOR_DATA_SIZE);

    // The root node
    _flattenChildren(&(octree[_depth - 1u].at(0)), _depth - 1u);

    _volumeDimensions =
        Vector3ui(std::ceil((maxAABB.x - minAABB.x) / voxelSize), std::ceil((maxAABB.y - minAABB.y) / voxelSize),
                  std::ceil((maxAABB.z - minAABB.z) / voxelSize));
    _volumeSize = (uint32_t)_volumeDimensions.x * (uint32_t)_volumeDimensions.y * (uint32_t)_volumeDimensions.z;
}

VectorOctree::~VectorOctree() {}

void VectorOctree::_flattenChildren(VectorOctreeNode *node, uint32_t level)
{
    const std::vector<VectorOctreeNode *> children = node->getChildren();
    const auto &position = node->getCenter();
    const auto &direction = node->getValue();
    if ((children.empty()) || (level == 0))
    {
        _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_POSITION_X] = position.x;
        _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_POSITION_Y] = position.y;
        _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_POSITION_Z] = position.z;
        _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_DIRECTION_X] = direction.x;
        _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_DIRECTION_Y] = direction.y;
        _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_DIRECTION_Z] = direction.z;

        _offsetPerLevel[level] += 1u;
        return;
    }
    _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_POSITION_X] = position.x;
    _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_POSITION_Y] = position.y;
    _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_POSITION_Z] = position.z;
    _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_DIRECTION_X] = direction.x;
    _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_DIRECTION_Y] = direction.y;
    _flatData[_offsetPerLevel[level] * FIELD_VECTOR_DATA_SIZE + FIELD_VECTOR_OFFSET_DIRECTION_Z] = direction.z;

    _flatIndices[_offsetPerLevel[level] * 2u] = _offsetPerLevel[level - 1];
    _flatIndices[_offsetPerLevel[level] * 2u + 1] = _offsetPerLevel[level - 1] + children.size() - 1u;
    _offsetPerLevel[level] += 1u;

    for (VectorOctreeNode *child : children)
        _flattenChildren(child, level - 1u);
}
} // namespace core
