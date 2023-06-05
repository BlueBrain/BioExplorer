/*
 * Copyright (c) 2019, EPFL/Blue Brain Project
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

#include "../../CommonStructs.h"

namespace core
{
#define float3_as_args(u)                        \
    reinterpret_cast<unsigned int &>((u).x),     \
        reinterpret_cast<unsigned int &>((u).y), \
        reinterpret_cast<unsigned int &>((u).z)

extern "C"
{
    __constant__ Params params;
}

__forceinline__ __device__ float luminance(const float3 &rgb)
{
    const float3 ntsc_luminance = {0.30f, 0.59f, 0.11f};
    return dot(rgb, ntsc_luminance);
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

static __device__ __inline__ void setRadiancePRD(const RadiancePRD &prd)
{
    optixSetPayload_0(__float_as_uint(prd.result.x));
    optixSetPayload_1(__float_as_uint(prd.result.y));
    optixSetPayload_2(__float_as_uint(prd.result.z));
    optixSetPayload_3(__float_as_uint(prd.importance));
    optixSetPayload_4(prd.depth);
}

static __device__ __inline__ OcclusionPRD getOcclusionPRD()
{
    OcclusionPRD prd;
    prd.attenuation.x = __uint_as_float(optixGetPayload_0());
    prd.attenuation.y = __uint_as_float(optixGetPayload_1());
    prd.attenuation.z = __uint_as_float(optixGetPayload_2());
    return prd;
}

static __device__ __inline__ void setOcclusionPRD(const OcclusionPRD &prd)
{
    optixSetPayload_0(__float_as_uint(prd.attenuation.x));
    optixSetPayload_1(__float_as_uint(prd.attenuation.y));
    optixSetPayload_2(__float_as_uint(prd.attenuation.z));
}

static __device__ __inline__ float3 traceRadianceRay(float3 origin,
                                                     float3 direction,
                                                     int depth,
                                                     float importance)
{
    RadiancePRD prd;
    prd.depth = depth;
    prd.importance = importance;

    optixTrace(params.handle, origin, direction, params.scene_epsilon, 1e16f,
               0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
               RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE,
               float3_as_args(prd.result),
               /* Can't use __float_as_uint() because it returns rvalue but
                  payload requires a lvalue */
               reinterpret_cast<unsigned int &>(prd.importance),
               reinterpret_cast<unsigned int &>(prd.depth));

    return prd.result;
}

static __device__ void phongShadowed()
{
    // this material is opaque, so it fully attenuates all shadow rays
    OcclusionPRD prd = getOcclusionPRD();
    prd.attenuation = make_float3(0.f);
    setOcclusionPRD(prd);
}

static __device__ void phongShade(float3 p_Kd, float3 p_Ka, float3 p_Ks,
                                  float3 p_Kr, float p_phong_exp,
                                  float3 p_normal)
{
    RadiancePRD prd = getRadiancePRD();

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_t = optixGetRayTmax();

    float3 hit_point = ray_orig + ray_t * ray_dir;

    // ambient contribution
    float3 result = p_Ka * params.ambient_light_color;

    // compute direct lighting
    BasicLight light = params.light;
    float Ldist = length(light.pos - hit_point);
    float3 L = normalize(light.pos - hit_point);
    float nDl = dot(p_normal, L);

    // cast shadow ray
    float3 light_attenuation = make_float3(static_cast<float>(nDl > 0.0f));
    if (nDl > 0.0f)
    {
        OcclusionPRD shadow_prd;
        shadow_prd.attenuation = make_float3(1.0f);

        optixTrace(params.handle, hit_point, L, 0.01f, Ldist, 0.0f,
                   OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
                   RAY_TYPE_OCCLUSION, RAY_TYPE_COUNT, RAY_TYPE_OCCLUSION,
                   float3_as_args(shadow_prd.attenuation));

        light_attenuation = shadow_prd.attenuation;
    }

    // If not completely shadowed, light the hit point
    if (fmaxf(light_attenuation) > 0.0f)
    {
        float3 Lc = light.color * light_attenuation;

        result += p_Kd * nDl * Lc;

        float3 H = normalize(L - ray_dir);
        float nDh = dot(p_normal, H);
        if (nDh > 0)
        {
            float power = pow(nDh, p_phong_exp);
            result += p_Ks * power * Lc;
        }
    }

    if (fmaxf(p_Kr) > 0)
    {
        // ray tree attenuation
        float new_importance = prd.importance * luminance(p_Kr);
        int new_depth = prd.depth + 1;

        // reflection ray
        // compare new_depth to max_depth - 1 to leave room for a potential
        // shadow ray trace
        if (new_importance >= 0.01f && new_depth <= params.max_depth - 1)
        {
            float3 R = reflect(ray_dir, p_normal);

            result += p_Kr *
                      traceRadianceRay(hit_point, R, new_depth, new_importance);
        }
    }

    // pass the color back
    prd.result = result;
    setRadiancePRD(prd);
}

extern "C" __global__ void __closesthit__radiance()
{
    const HitGroupData *sbt_data = (HitGroupData *)optixGetSbtDataPointer();
    const Phong &phong = sbt_data->shading.phong;

    const float3 object_normal =
        make_float3(__uint_as_float(optixGetAttribute_0()),
                    __uint_as_float(optixGetAttribute_1()),
                    __uint_as_float(optixGetAttribute_2()));

    const float3 world_normal =
        normalize(optixTransformNormalFromObjectToWorldSpace(object_normal));
    const float3 normal =
        faceforward(world_normal, -optixGetWorldRayDirection(), world_normal);
    phongShade(phong.Kd, phong.Ka, phong.Ks, phong.Kr, phong.phong_exp, normal);
}

extern "C" __global__ void __closesthit__full_occlusion()
{
    phongShadowed();
}
} // namespace core
