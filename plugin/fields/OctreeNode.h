/* Copyright (c) 2015-2017, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *                     Grigori Chevtchenko <grigori.chevtchenko@epfl.ch>
 *
 * This file is part of SIMDVoxelizer <https://github.com/favreau/SIMDVoxelizer>
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

#ifndef BIOEXPLORER_OCTREENODE_H
#define BIOEXPLORER_OCTREENODE_H

#include <glm/glm.hpp>
#include <vector>

class OctreeNode
{
public:
    OctreeNode(const glm::vec3 center, const float size);

    void addValue(const float value);
    void setChild(OctreeNode* child);

    const std::vector<OctreeNode*>& getChildren() const;

    const glm::vec3& getCenter() const;
    float getValue() const;

private:
    float _value;

    glm::vec3 _center;
    glm::vec3 _size;

    std::vector<OctreeNode*> _children;
};

#endif // BIOEXPLORER_OCTREENODE_H
