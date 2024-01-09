/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "Field.h"
#include "Field_ispc.h"

#include <platform/engines/ospray/OSPRayProperties.h>

#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/common/Model.h>
#include <ospray/SDK/volume/Volume.h>

namespace core
{
namespace engine
{
namespace ospray
{
Field::Field()
{
    this->ispcEquivalent = ispc::Field_createInstance(this, OSP_FLOAT, (const ispc::vec3i &)this->_dimensions);
}

void Field::commit()
{
    _indices = getParamData(OSPRAY_VOLUME_OCTREE_INDICES, nullptr);
    _values = getParamData(OSPRAY_VOLUME_OCTREE_VALUES, nullptr);
    _dimensions = getParam3i(OSPRAY_VOLUME_DIMENSIONS, ::ospray::vec3ui());
    _spacing = getParam3f(OSPRAY_VOLUME_SPACING, ::ospray::vec3f());
    _offset = getParam3f(OSPRAY_VOLUME_OFFSET, ::ospray::vec3f());

    ::ispc::Field_set(getIE(), (ispc::vec3i &)_dimensions, (ispc::vec3f &)_spacing, (ispc::vec3f &)_offset,
                      _indices->data, _values->data);
}

int Field::setRegion(const void *source, const ::ospray::vec3i &index, const ::ospray::vec3i &count)
{
    return true;
}

OSP_REGISTER_VOLUME(Field, octree_volume);
} // namespace ospray
} // namespace engine
} // namespace core
