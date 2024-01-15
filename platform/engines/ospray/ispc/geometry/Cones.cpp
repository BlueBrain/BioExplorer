/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Author: Jafet Villafranca Diaz <jafet.villafrancadiaz@epfl.ch>
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

#include "Cones.h"
#include "Cones_ispc.h"

#include <platform/core/common/geometry/Cone.h>
#include <platform/engines/ospray/OSPRayProperties.h>

#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/common/Model.h>

#include <climits>

using namespace core;

namespace core
{
namespace engine
{
namespace ospray
{
Cones::Cones()
{
    this->ispcEquivalent = ::ispc::Cones_create(this);
}

void Cones::finalize(::ospray::Model* model)
{
    data = getParamData(OSPRAY_GEOMETRY_PROPERTY_CONES, nullptr);
    constexpr size_t bytesPerCone = sizeof(Cone);

    if (data.ptr == nullptr || bytesPerCone == 0)
        throw std::runtime_error("#ospray:geometry/cones: no 'cones' data specified");

    const size_t numCones = data->numBytes / bytesPerCone;
    ::ispc::ConesGeometry_set(getIE(), model->getIE(), data->data, numCones);
}

OSP_REGISTER_GEOMETRY(Cones, cones);
} // namespace ospray
} // namespace engine
} // namespace core
