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

#include <brayns/common/types.h>

namespace bioexplorer
{
namespace fields
{
/**
 * @brief The OctreeNode class implement a spherical node of the Octree
 * acceleration structure used by the Fields renderer
 *
 */
class OctreeNode
{
public:
    /**
     * @brief Construct a new Octree Node object
     *
     * @param The center of the node
     * @param The node size
     */
    OctreeNode(const glm::vec3 center, const float size);

    /**
     * @brief Add a value to the node
     *
     * @param The value of the node
     */
    void addValue(const float value);

    /**
     * @brief Add a Child to the node
     *
     * @param The node child
     */
    void setChild(OctreeNode* child);

    /**
     * @brief Get the node children
     *
     * @return A vector of nodes
     */
    const std::vector<OctreeNode*>& getChildren() const;

    /**
     * @brief Get the center of the node
     *
     * @return The center of the node
     */
    const glm::vec3& getCenter() const;

    /**
     * @brief Get the value of the node
     *
     * @return The value of the node
     */
    const float getValue() const;

private:
    float _value;

    glm::vec3 _center;
    glm::vec3 _size;

    std::vector<OctreeNode*> _children;
};
} // namespace fields
} // namespace bioexplorer