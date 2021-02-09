/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#include <glm/glm.hpp>
#include <stdint.h>
#include <vector>

#include "OctreeNode.h"

namespace bioexplorer
{
class Octree
{
public:
    Octree(const std::vector<float> &events, float voxelSize,
           const glm::vec3 &minAABB, const glm::vec3 &maxAABB);
    ~Octree();

    const glm::uvec3 &getVolumeDim() const;
    uint64_t getVolumeSize() const;
    uint32_t getOctreeSize() const;

    const std::vector<uint32_t> &getFlatIndexes() const;
    const std::vector<float> &getFlatData() const;

private:
    void _flattenChildren(const OctreeNode *node, uint32_t level);

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

    glm::uvec3 _volumeDim;
    uint64_t _volumeSize;
    uint32_t _octreeSize;

    uint32_t *_offsetPerLevel;

    std::vector<uint32_t> _flatIndexes;
    std::vector<float> _flatData;
};
}