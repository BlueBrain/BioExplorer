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

#include "Node.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/UniqueId.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>

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
    {
        brayns::PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                           static_cast<int>(MaterialShadingMode::basic)});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        props.setProperty({MATERIAL_PROPERTY_CHAMELEON_MODE,
                           static_cast<int>(MaterialChameleonMode::receiver)});
        props.setProperty({MATERIAL_PROPERTY_NODE_ID, static_cast<int>(_uuid)});
        material.second->updateProperties(props);
    }
}

} // namespace common
} // namespace bioexplorer
