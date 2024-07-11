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

#include "VectorOctreeNode.h"

namespace core
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
} // namespace core
