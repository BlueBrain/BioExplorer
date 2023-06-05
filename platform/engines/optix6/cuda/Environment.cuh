/* Copyright (c) 2019, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Core <https://github.com/BlueBrain/Core>
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

#include <optix_world.h>

#include "Helpers.cuh"

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float3, bgColor, , );
rtDeclareVariable(int, envmap, , );
rtDeclareVariable(uint, use_envmap, , );
rtDeclareVariable(uint, showBackground, , );

static __device__ inline float3 getEnvironmentColor()
{
    if (showBackground)
    {
        if (use_envmap)
        {
            const float2 uv = getEquirectangularUV(ray.direction);
            return linearToSRGB(tonemap(make_float3(optix::rtTex2D<float4>(envmap, uv.x, uv.y))));
        }
        return bgColor;
    }
    return make_float3(0.f);
}
