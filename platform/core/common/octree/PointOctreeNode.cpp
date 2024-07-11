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
