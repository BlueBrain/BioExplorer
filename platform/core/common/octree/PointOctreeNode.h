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

namespace core
{
/**
 * @brief The PointOctreeNode class implement a spherical node of the Octree
 * acceleration structure used by the Fields renderer
 *
 */
class PointOctreeNode
{
public:
    /**
     * @brief Construct a new Octree Node object
     *
     * @param The center of the node
     * @param The node size
     */
    PointOctreeNode(const Vector3f& center, const double size);

    /**
     * @brief Add a value to the node
     *
     * @param The value of the node
     */
    void addValue(const double value);

    /**
     * @brief Add a Child to the node
     *
     * @param The node child
     */
    void setChild(PointOctreeNode* child);

    /**
     * @brief Get the node children
     *
     * @return A vector of nodes
     */
    const std::vector<PointOctreeNode*>& getChildren() const;

    /**
     * @brief Get the center of the node
     *
     * @return The center of the node
     */
    const Vector3f& getCenter() const;

    /**
     * @brief Get the value of the node
     *
     * @return The value of the node
     */
    double getValue() const;

private:
    double _value;

    Vector3f _center;
    Vector3f _size;

    std::vector<PointOctreeNode*> _children;
};
} // namespace core
