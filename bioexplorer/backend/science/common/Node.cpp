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

#include "Node.h"

#include <science/common/UniqueId.h>

#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>

using namespace core;

namespace bioexplorer
{
namespace common
{
Node::Node(const Vector3d& scale)
    : _scale(scale)
{
    // Unique ID
    _uuid = UniqueId::get();
}

const ModelDescriptorPtr Node::getModelDescriptor() const
{
    return _modelDescriptor;
}

void Node::_setMaterialExtraAttributes()
{
    auto materials = _modelDescriptor->getModel().getMaterials();
    for (auto& material : materials)
        material.second->setNodeId(_uuid);
}

} // namespace common
} // namespace bioexplorer
