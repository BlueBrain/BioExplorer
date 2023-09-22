/*
 * Copyright (c) 2019-2023, EPFL/Blue Brain Project
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

#include <platform/engines/optix6/cuda/Environment.cuh>
#include <platform/engines/optix6/cuda/Helpers.cuh>
#include <platform/engines/optix6/cuda/Random.cuh>
#include <platform/engines/optix6/cuda/renderer/Volume.cuh>

#include <platform/core/common/CommonTypes.h>

static __device__ inline void shade(bool textured)
{
    const float3 world_shading_normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    const float3 world_geometric_normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    const float3 normal = optix::faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

    float3 color;
    if (textured && albedoMetallic_map)
        color = make_float3(optix::rtTex2D<float4>(albedoMetallic_map, texcoord.x, texcoord.y));
    else
        color = Kd;

    prd.result = make_float4(color * max(0.f, optix::dot(-ray.direction, normal)), 1.f);
}

RT_PROGRAM void any_hit_shadow()
{
    rtTerminateRay();
}

RT_PROGRAM void closest_hit_radiance()
{
    shade(false);
}

RT_PROGRAM void closest_hit_radiance_textured()
{
    shade(true);
}

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_color(bad_color);
}
