/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include <plugin/common/Types.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;

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
    Node(const Vector3d& scale = Vector3d(1.0, 1.0, 1.0));

    /**
     * @brief Get the Model Descriptor object
     *
     * @return ModelDescriptorPtr Pointer to the model descriptor
     */
    const ModelDescriptorPtr getModelDescriptor() const;

protected:
    void _setMaterialExtraAttributes();

    ModelDescriptorPtr _modelDescriptor{nullptr};
    Boxd _bounds;
    uint32_t _uuid;
    Vector3d _scale;
};

typedef std::shared_ptr<Node> NodePtr;
typedef std::map<std::string, NodePtr> NodeMap;

} // namespace common
} // namespace bioexplorer