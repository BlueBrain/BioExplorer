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
