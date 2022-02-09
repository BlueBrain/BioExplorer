/* Copyright (c) 2015-2022, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *                     Grigori Chevtchenko <grigori.chevtchenko@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include <glm/glm.hpp>
#include <stdint.h>
#include <vector>

#include "OctreeNode.h"

namespace fieldrenderer
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
} // namespace fieldrenderer