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

// #include "Helpers.h"
#include <platform/engines/optix6/cuda/Context.cuh>
#include <platform/engines/optix6/cuda/Helpers.cuh>
#include <platform/engines/optix6/cuda/Random.cuh>

#include <optixu/optixu_matrix_namespace.h>

static const float OPENDECK_RADIUS = 2.55f;
static const float OPENDECK_HEIGHT = 2.3f;
static const float OPENDECK_METALSTRIPE_HEIGHT = 0.045f;
static const float PI = 3.141592f;
static const float OPENDECK_BEZEL_ANGLE = PI / 180.0f * 7.98995f;
static const float ANGLE_PER_BORDER_SEGMENT = (PI - 8.0f * OPENDECK_BEZEL_ANGLE) / 7.0f + OPENDECK_BEZEL_ANGLE;
static const float FULL_ANGLE = ANGLE_PER_BORDER_SEGMENT + OPENDECK_BEZEL_ANGLE;

using namespace optix;

rtDeclareVariable(uint, segmentID, , ); // even segmentsID are right eye
                                                // buffers and odd are left eye
                                                // buffers
rtDeclareVariable(float3, headPos, , );
rtDeclareVariable(float3, headUVec, , );

rtDeclareVariable(float, half_ipd, , );

// Pass 'seed' by reference to keep randomness state
__device__ float4 launch(uint& seed, const float2 screen, const bool use_randomness)
{
    float eyeDelta = 0.0f;
    float alpha = 0.0f;
    float3 dPx, dPy;

    float2 sample = make_float2(launch_index) / screen;

    if (segmentID <= 13 && segmentID % 2 == 0)
    {
        eyeDelta = half_ipd;
        uint angularOffset = segmentID / 2;

        if (segmentID == 0)
            alpha = sample.x * FULL_ANGLE;
        else if (segmentID == 12)
            alpha = PI - FULL_ANGLE + sample.x * FULL_ANGLE;
        else
            alpha = angularOffset * (FULL_ANGLE - OPENDECK_BEZEL_ANGLE) + sample.x * FULL_ANGLE;
    }
    else if (segmentID <= 13 && segmentID % 2 == 1)
    {
        eyeDelta = -half_ipd;
        uint angularOffset = segmentID / 2;
        if (segmentID == 1)
            alpha = sample.x * FULL_ANGLE;
        else if (segmentID == 13)
            alpha = PI - FULL_ANGLE + sample.x * FULL_ANGLE;
        else
            alpha = angularOffset * (FULL_ANGLE - OPENDECK_BEZEL_ANGLE) + sample.x * FULL_ANGLE;
    }
    else if (segmentID == 14)
    {
        eyeDelta = half_ipd;
    }
    else if (segmentID == 15)
    {
        eyeDelta = -half_ipd;
    }

    float3 pixelPos;
    if (segmentID <= 13)
    {
        pixelPos.x = OPENDECK_RADIUS * -cosf(alpha);
        pixelPos.y = OPENDECK_METALSTRIPE_HEIGHT + OPENDECK_HEIGHT * sample.y;
        pixelPos.z = OPENDECK_RADIUS * -sinf(alpha);

        dPx =
            make_float3(FULL_ANGLE * OPENDECK_RADIUS * sinf(alpha), 0.0f, FULL_ANGLE * OPENDECK_RADIUS * -cosf(alpha));
        dPy = make_float3(0.0f, OPENDECK_HEIGHT, 0.0f);
    }
    else if (segmentID > 13)
    {
        pixelPos.x = 2.0f * OPENDECK_RADIUS * (sample.x - 0.5f);
        pixelPos.y = 0.0f;
        pixelPos.z = -OPENDECK_RADIUS * sample.y;

        dPx = make_float3(2.0f * OPENDECK_RADIUS, 0.0f, 0.0f);
        dPy = make_float3(0.0f, 0.0f, -OPENDECK_RADIUS);
    }

    // The tracking model of the 3d glasses is inversed
    // so we need to negate CamDu here.
    const float3 eyeDeltaPos = -headUVec * eyeDelta;

    optix::Matrix3x3 transform;
    transform.setCol(0, U);
    transform.setCol(1, V);
    transform.setCol(2, W);

    const float3 d = pixelPos - headPos + eyeDeltaPos;
    const float dotD = dot(d, d);
    const float denom = pow(dotD, 1.5f);
    float3 dir = normalize(d);

    float3 dirDx = (dotD * dPx - dot(d, dPx) * d) / (denom * screen.x);
    float3 dirDy = (dotD * dPy - dot(d, dPy) * d) / (denom * screen.y);

    PerRayData_radiance prd;
    prd.importance = 1.f;
    prd.depth = 0;
    prd.rayDdx = transform * dirDx;
    prd.rayDdy = transform * dirDy;
    dir = transform * dir;

    const float3 org = eye + headPos - eyeDeltaPos;
    float near = sceneEpsilon;
    float far = INFINITY;

    applyClippingPlanes(org, dir, near, far);
    optix::Ray ray(org, dir, radiance_ray_type, near, far);

    rtTrace(top_object, ray, prd);

    return make_float4(make_float3(prd.result) * mainExposure, prd.result.w);
}

RT_PROGRAM void openDeckCamera()
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
}
