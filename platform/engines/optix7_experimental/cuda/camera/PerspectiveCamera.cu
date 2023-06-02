/* Copyright (c) 2019, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include <optix.h>

#include "../../CommonStructs.h"

#include <cuda/helpers.h>

#include <sutil/vec_math.h>

namespace core
{
extern "C"
{
    __constant__ Params params;
}

static __forceinline__ __device__ void trace(OptixTraversableHandle handle,
                                             float3 ray_origin,
                                             float3 ray_direction, float tmin,
                                             float tmax, float3* prd)
{
    unsigned int p0, p1, p2;
    p0 = __float_as_uint(prd->x);
    p1 = __float_as_uint(prd->y);
    p2 = __float_as_uint(prd->z);
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
               0.0f, // rayTime
               OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
               0, // SBT offset
               0, // SBT stride
               0, // missSBTIndex
               p0, p1, p2);
    prd->x = __uint_as_float(p0);
    prd->y = __uint_as_float(p1);
    prd->z = __uint_as_float(p2);
}

static __forceinline__ __device__ void setPayload(float3 p)
{
    optixSetPayload_0(__float_as_uint(p.x));
    optixSetPayload_1(__float_as_uint(p.y));
    optixSetPayload_2(__float_as_uint(p.z));
}

static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(__uint_as_float(optixGetPayload_0()),
                       __uint_as_float(optixGetPayload_1()),
                       __uint_as_float(optixGetPayload_2()));
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    const float3 U = rtData->camera_u;
    const float3 V = rtData->camera_v;
    const float3 W = rtData->camera_w;
    const float2 d =
        2.0f *
            make_float2(static_cast<float>(idx.x) / static_cast<float>(dim.x),
                        static_cast<float>(idx.y) / static_cast<float>(dim.y)) -
        1.0f;

    const float3 origin = rtData->cam_eye;
    const float3 direction = normalize(d.x * U + d.y * V + W);
    float3 payload_rgb = make_float3(0.5f, 0.5f, 0.5f);
    trace(params.handle, origin, direction,
          0.00f, // tmin
          1e16f, // tmax
          &payload_rgb);

    params.frame_buffer[idx.y * params.width + idx.x] = make_color(payload_rgb);
}

static __device__ __inline__ RadiancePRD getRadiancePRD()
{
    RadiancePRD prd;
    prd.result.x = __uint_as_float(optixGetPayload_0());
    prd.result.y = __uint_as_float(optixGetPayload_1());
    prd.result.z = __uint_as_float(optixGetPayload_2());
    prd.importance = __uint_as_float(optixGetPayload_3());
    prd.depth = optixGetPayload_4();
    return prd;
}

static __device__ __inline__ void setRadiancePRD(const RadiancePRD& prd)
{
    optixSetPayload_0(__float_as_uint(prd.result.x));
    optixSetPayload_1(__float_as_uint(prd.result.y));
    optixSetPayload_2(__float_as_uint(prd.result.z));
    optixSetPayload_3(__float_as_uint(prd.importance));
    optixSetPayload_4(prd.depth);
}

extern "C" __global__ void __miss__constant_bg()
{
    const MissData* sbt_data = (MissData*)optixGetSbtDataPointer();
    RadiancePRD prd = getRadiancePRD();
#if 1
    prd.result = sbt_data->bg_color;
#else
    // This is to test the camera and the frame buffer
    const float3 ray_dir = optixGetWorldRayDirection();
    prd.result = sbt_data->bg_color * 0.75 + 0.25 * normalize(ray_dir);
#endif
    setRadiancePRD(prd);
}

extern "C" __global__ void __closesthit__ch()
{
    float t_hit = optixGetRayTmax();
    // Backface hit not used.
    // float  t_hit2 = __uint_as_float( optixGetAttribute_0() );

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();

    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int sbtGASIndex = optixGetSbtGASIndex();

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData(gas, prim_idx, sbtGASIndex, 0.f, &q);

    float3 world_raypos = ray_orig + t_hit * ray_dir;
    float3 obj_raypos = optixTransformPointFromWorldToObjectSpace(world_raypos);
    float3 obj_normal = (obj_raypos - make_float3(q)) / q.w;
    float3 world_normal =
        normalize(optixTransformNormalFromObjectToWorldSpace(obj_normal));

    setPayload(world_normal * 0.5f + 0.5f);
}
} // namespace core
