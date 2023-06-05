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

#include "../../OptiXCommonStructs.h"

#include "../Helpers.cuh"
#include "../Random.cuh"

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, dir, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(float, scene_epsilon, , );
rtBuffer<uchar4, 2> output_buffer;
rtBuffer<float4, 2> accum_buffer;
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, frame, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtDeclareVariable(float, height, , );
rtDeclareVariable(float4, jitter4, , );
rtDeclareVariable(unsigned int, samples_per_pixel, , );

rtBuffer<float4, 1> clip_planes;
rtDeclareVariable(unsigned int, nb_clip_planes, , );

__device__ void getClippingValues(const float3& ray_origin, const float3& ray_direction, float& near, float& far)
{
    for (int i = 0; i < nb_clip_planes; ++i)
    {
        float4 clipPlane = clip_planes[i];
        const float3 planeNormal = {clipPlane.x, clipPlane.y, clipPlane.z};
        float rn = dot(ray_direction, planeNormal);
        if (rn == 0.f)
            rn = scene_epsilon;
        float d = clipPlane.w;
        float t = -(dot(planeNormal, ray_origin) + d) / rn;
        if (rn > 0.f) // opposite direction plane
            near = max(near, t);
        else
            far = min(far, t);
    }
}

// Pass 'seed' by reference to keep randomness state
__device__ float3 launch(unsigned int& seed, const float2 screen, const bool use_randomness)
{
    // Subpixel jitter: send the ray through a different position inside the
    // pixel each time, to provide antialiasing.
    float2 subpixel_jitter = use_randomness ? make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f) : make_float2(0.f, 0.f);

    float2 p = (make_float2(launch_index) + subpixel_jitter) / screen * 2.f - 1.f;

    const float3 ray_origin = W + screen.x * U + screen.y * V;
    const float3 ray_direction = optix::normalize(dir);

    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.rayDdx = screen.x * U;
    prd.rayDdy = screen.y * V;

    // lens sampling
    float2 sample = optix::square_to_disk(make_float2(jitter4.z, jitter4.w));

    float near = scene_epsilon;
    float far = INFINITY;

    getClippingValues(ray_origin, ray_direction, near, far);

    optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, near, far);

    rtTrace(top_object, ray, prd);

    return prd.result;
}

RT_PROGRAM void orthographicCamera()
{
    const size_t2 screen = output_buffer.size();
    const float2 screen_f = make_float2(screen);

    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    const int num_samples = max(1, samples_per_pixel);
    // We enable randomness if we are using subpixel sampling or accumulation
    const bool use_randomness = frame > 0 || num_samples > 1;

    float3 result = make_float3(0, 0, 0);
    for (int i = 0; i < num_samples; i++)
        result += launch(seed, screen_f, use_randomness);
    result /= num_samples;

    float4 acc_val;
    if (frame > 0)
    {
        acc_val = accum_buffer[launch_index];
        acc_val = lerp(acc_val, make_float4(result, 0.f), 1.0f / static_cast<float>(frame + 1));
    }
    else
        acc_val = make_float4(result, 1.f);

    output_buffer[launch_index] = make_color(make_float3(acc_val));

    if (accum_buffer.size().x > 1 && accum_buffer.size().y > 1)
        accum_buffer[launch_index] = acc_val;
}

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_color(bad_color);
}
