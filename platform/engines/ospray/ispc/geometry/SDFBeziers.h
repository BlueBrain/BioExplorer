/*
 * Copyright (c) 2019, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Author: Sebastien Speierer <sebastien.speierer@epfl.ch>
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

#pragma once

#include "ospray/SDK/geometry/Geometry.h"
#include <platform/core/common/Types.h>

namespace core
{
namespace engine
{
namespace ospray
{
struct SDFBeziers : public ::ospray::Geometry
{
    SDFBeziers();

    std::string toString() const final { return "SDFBeziers"; }
    void finalize(::ospray::Model* model) final;

    ::ospray::Ref<::ospray::Data> data;
    float epsilon;
    ::ospray::uint64 nbMarchIterations;
    float blendFactor;
    float blendLerpFactor;
    float omega;
    float distance;
};
} // namespace ospray
} // namespace engine
} // namespace core