/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include "CommonDefines.h"

#include <Defines.h>

#if __cplusplus
namespace core
{
#endif
struct Cylinder
{
#if __cplusplus
    Cylinder(const Vector3f c = {0.f, 0.f, 0.f}, const Vector3f u = {0.f, 0.f, 0.f}, const float r = 0.f,
             const uint64_t data = 0)
        : userData(data)
        , center(c)
        , up(u)
        , radius(r)
    {
    }
#endif

    UINT64_T userData;
    VEC3_TYPE center;
    VEC3_TYPE up;
    float radius;
    __MEMORY_ALIGNMENT__
};

#if __cplusplus
} // Platform
#endif
