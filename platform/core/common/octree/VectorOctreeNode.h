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

#pragma once

#include <platform/core/common/Types.h>

namespace core
{
/**
 * @brief The VectorOctreeNode class implement a spherical node of the Octree
 * acceleration structure used by the Fields renderer
 *
 */
class VectorOctreeNode
{
public:
    /**
     * @brief Construct a new Octree Node object
     *
     * @param The center of the node
     * @param The node size
     */
    VectorOctreeNode(const Vector3f& center, const double size);

    /**
     * @brief Add a value to the node
     *
     * @param The value of the node
     */
    void addValue(const Vector3d& vector);

    /**
     * @brief Add a Child to the node
     *
     * @param The node child
     */
    void setChild(VectorOctreeNode* child);

    /**
     * @brief Get the node children
     *
     * @return A vector of nodes
     */
    const std::vector<VectorOctreeNode*>& getChildren() const;

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
    const Vector3d& getValue() const;

private:
    Vector3d _value;
    Vector3f _center;
    Vector3f _size;

    std::vector<VectorOctreeNode*> _children;
};
} // namespace core
