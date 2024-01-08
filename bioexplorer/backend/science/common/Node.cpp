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
