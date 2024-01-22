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

#include <platform/engines/optix6/cuda/Context.cuh>
#include <platform/engines/optix6/cuda/Helpers.cuh>
#include <platform/engines/optix6/cuda/Random.cuh>

using namespace optix;

// Pass 'seed' by reference to keep randomness state
__device__ float4 launch(uint& seed, const float2 screen, const bool use_randomness)
{
    // Subpixel jitter: send the ray through a different position inside the
    // pixel each time, to provide anti-aliasing.
    const float2 subpixel_jitter =
        use_randomness ? make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f) : make_float2(0.f, 0.f);

    // Normalized pixel position (from -0.5 to 0.5)
    const float2 p = (make_float2(launch_index) + subpixel_jitter) / screen * 2.f - 1.f;

    const float3 ray_origin = W + p.x * U + p.y * V;
    const float3 ray_direction = dir;

    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.zDepth = INFINITY;

    float near = sceneEpsilon;
    float far = INFINITY;

    // Clipping planes
    if (enableClippingPlanes)
        applyClippingPlanes(ray_origin, ray_direction, near, far);

    const optix::Ray ray(ray_origin, ray_direction, radiance_ray_type, near, far);
    rtTrace(top_object, ray, prd);

    return make_float4(make_float3(prd.result) * mainExposure, prd.result.w);
}

RT_PROGRAM void orthographicCamera()
{
    const size_t2 screen = output_buffer.size();
    const float2 screen_f = make_float2(screen);

    uint seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    const int num_samples = max(1, samples_per_pixel);
    // We enable randomness if we are using subpixel sampling or accumulation
    const bool use_randomness = frame > 0 || num_samples > 1;
    float4 result = make_float4(0.f);
    for (int i = 0; i < num_samples; i++)
        result += launch(seed, screen_f, use_randomness);
    result /= num_samples;

    float4 acc_val;
    if (frame > 0)
    {
        acc_val = accum_buffer[launch_index];
        acc_val = lerp(acc_val, result, 1.0f / static_cast<float>(frame + 1));
    }
    else
        acc_val = result;

    output_buffer[launch_index] = make_color(acc_val);

    if (accum_buffer.size().x > 1 && accum_buffer.size().y > 1)
        accum_buffer[launch_index] = acc_val;
}

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_color(bad_color);
    depth_buffer[launch_index] = INFINITY;
}
