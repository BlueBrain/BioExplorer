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

#include "VectorOctreeNode.h"

using namespace core;

namespace bioexplorer
{
namespace common
{
VectorOctreeNode::VectorOctreeNode(const Vector3f& center, const double size)
    : _center(center)
    , _size(size)
{
}

void VectorOctreeNode::setChild(VectorOctreeNode* child)
{
    _children.push_back(child);
}

void VectorOctreeNode::addValue(const core::Vector3d& value)
{
    _value += value;
}

const Vector3f& VectorOctreeNode::getCenter() const
{
    return _center;
}

const core::Vector3d& VectorOctreeNode::getValue() const
{
    return _value;
}

const std::vector<VectorOctreeNode*>& VectorOctreeNode::getChildren() const
{
    return _children;
}
} // namespace common
} // namespace bioexplorer