/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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

#include "PointOctreeNode.h"

namespace core
{
PointOctreeNode::PointOctreeNode(const Vector3f& center, const double size)
    : _center(center)
    , _size(size)
{
}

void PointOctreeNode::setChild(PointOctreeNode* child)
{
    _children.push_back(child);
}

void PointOctreeNode::addValue(const double value)
{
    if (value > _value)
        _value = value;
}

const Vector3f& PointOctreeNode::getCenter() const
{
    return _center;
}

double PointOctreeNode::getValue() const
{
    return _value;
}

const std::vector<PointOctreeNode*>& PointOctreeNode::getChildren() const
{
    return _children;
}
} // namespace core
