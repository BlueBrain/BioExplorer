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

#include <science/common/Types.h>

namespace bioexplorer
{
namespace common
{
/**
 * @brief The Node class
 */
class Node
{
public:
    /**
     * @brief Construct a new Node object
     *
     */
    Node(const core::Vector3d& scale = core::Vector3d(1.0, 1.0, 1.0));

    /**
     * @brief Destroy the Node object
     *
     */
    virtual ~Node() = default;

    /**
     * @brief Get the Model Descriptor object
     *
     * @return ModelDescriptorPtr Pointer to the model descriptor
     */
    const core::ModelDescriptorPtr getModelDescriptor() const;

    /**
     * @brief Get the bounds of the node
     *
     * @return const Boxf& Bounds of the node
     */
    const core::Boxd& getBounds() const { return _bounds; };

protected:
    void _setMaterialExtraAttributes();

    core::ModelDescriptorPtr _modelDescriptor{nullptr};
    core::Boxd _bounds;
    uint32_t _uuid;
    core::Vector3d _scale;
};

typedef std::shared_ptr<Node> NodePtr;
typedef std::map<std::string, NodePtr> NodeMap;

} // namespace common
} // namespace bioexplorer