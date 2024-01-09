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

#include "Fields.h"
#include "Fields_ispc.h"

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/OSPRayProperties.h>

#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/common/Model.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

namespace core
{
namespace engine
{
namespace ospray
{
Fields::Fields()
{
    this->ispcEquivalent = ispc::Field_create(this);
}

void Fields::finalize(::ospray::Model *model)
{
    const size_t numFields = 1;
    _indices = getParamData(OSPRAY_GEOMETRY_PROPERTY_FIELD_INDICES, nullptr);
    _values = getParamData(OSPRAY_GEOMETRY_PROPERTY_FIELD_VALUES, nullptr);

    _dimensions = getParam3i(OSPRAY_GEOMETRY_PROPERTY_FIELD_DIMENSIONS, ::ospray::vec3i());
    _spacing = getParam3f(OSPRAY_GEOMETRY_PROPERTY_FIELD_SPACING, ::ospray::vec3f());
    _offset = getParam3f(OSPRAY_GEOMETRY_PROPERTY_FIELD_OFFSET, ::ospray::vec3f());

    ::ispc::Field_set(getIE(), model->getIE(), (ispc::vec3i &)_dimensions, (ispc::vec3f &)_spacing,
                      (ispc::vec3f &)_offset, _indices->data, _values->data, numFields);

    // Transfer function
    ::ospray::TransferFunction *transferFunction =
        (::ospray::TransferFunction *)getParamObject(DEFAULT_COMMON_TRANSFER_FUNCTION, nullptr);
    if (transferFunction)
        ::ispc::Field_setTransferFunction(getIE(), transferFunction->getIE());
}

OSP_REGISTER_GEOMETRY(Fields, fields);
} // namespace ospray
} // namespace engine
} // namespace core
