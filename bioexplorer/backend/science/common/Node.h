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