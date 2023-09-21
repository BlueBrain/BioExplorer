/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 /*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <platform/engines/optix6/cuda/Random.cuh>
#include <platform/engines/optix6/cuda/renderer/Volume.cuh>

using namespace optix;

const uint OFFSET_DIMENSIONS = 0;
const uint OFFSET_POSITION = OFFSET_DIMENSIONS + 3;
const uint OFFSET_SPACING = OFFSET_POSITION + 3;
const uint OFFSET_VOLUME_TEXTURE_SAMPLER_ID = OFFSET_SPACING + 3;
const uint OFFSET_TRANSFER_FUNCTION_TEXTURE_SAMPLER_ID = OFFSET_VOLUME_TEXTURE_SAMPLER_ID + 1;
const uint OFFSET_VALUE_RANGE = OFFSET_TRANSFER_FUNCTION_TEXTURE_SAMPLER_ID + 1;

rtDeclareVariable(uint, volume_size, , );

rtBuffer<float> volumes;

template <bool use_robust_method>
static __device__ void intersect_volume(int primIdx)
{
    const int idx = primIdx * volume_size;
    const float3 dimensions = {volumes[idx + OFFSET_DIMENSIONS], volumes[idx + OFFSET_DIMENSIONS + 1],
                               volumes[idx + OFFSET_DIMENSIONS + 2]};
    const float3 position = {volumes[idx + OFFSET_POSITION], volumes[idx + OFFSET_POSITION + 1],
                             volumes[idx + OFFSET_POSITION + 2]};
    const float3 spacing = {volumes[idx + OFFSET_SPACING], volumes[idx + OFFSET_SPACING + 1],
                            volumes[idx + OFFSET_SPACING + 2]};
    const int volumeSamplerId = static_cast<int>(volumes[idx + OFFSET_VOLUME_TEXTURE_SAMPLER_ID]);
    const int transferFunctionSamplerId = static_cast<int>(volumes[idx + OFFSET_TRANSFER_FUNCTION_TEXTURE_SAMPLER_ID]);
    const float2 valueRange = {volumes[idx + OFFSET_VALUE_RANGE], volumes[idx + OFFSET_VALUE_RANGE + 1]};

    const float3 boxMin = position;
    const float3 boxMax = position + dimensions * spacing;

    const float3 a = (boxMin - ray.origin) / ray.direction;
    const float3 b = (boxMax - ray.origin) / ray.direction;
    const float3 near = fminf(a, b);
    const float3 far = fmaxf(a, b);
    float t0 = fmaxf(near);
    float t1 = fminf(far);

    const ::optix::size_t2 screen = output_buffer.size();
    uint seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    const float diag = min(spacing.x, min(spacing.y, spacing.z));
    const float step = diag / volumeSamplingRate;
    const float random = rnd(seed) * step * 5.f;

    // Apply ray clipping
    t0 = max(t0, ray.tmin);
    t1 = min(t1, ray.tmax);

    if (t0 > 0.f && t0 <= t1)
    {
        float t = t0 + random;
        while (t < t1)
        {
            const float3 p0 = ((ray.origin + t * ray.direction) - position) / (spacing * dimensions);
            const float voxelValue = optix::rtTex3D<float>(volumeSamplerId, p0.x, p0.y, p0.z);
            const float4 voxelColor = calcTransferFunctionColor(transferFunctionSamplerId, valueRange, voxelValue);
            if (voxelColor.w > 0.f)
                if (rtPotentialIntersection(t - sceneEpsilon))
                {
                    float3 normal = make_float3(0.f);
                    if (volumeGradientShadingEnabled)
                    {
                        const float3 positions[6] = {{-1, 0, 0}, {1, 0, 0},  {0, -1, 0},
                                                     {0, 1, 0},  {0, 0, -1}, {0, 0, 1}};
                        for (const auto& position : positions)
                        {
                            const float3 p1 = p0 + (position * DEFAULT_GRADIENT_OFFSET);
                            const float v = optix::rtTex3D<float>(volumeSamplerId, p1.x, p1.y, p1.z);
                            normal += v * position;
                        }
                        normal = ::optix::normalize(-1.f * normal);
                    }
                    else
                        normal = make_float3(0, 1, 0);

                    geometric_normal = shading_normal = normal;
                    userDataIndex = 0;
                    texcoord = make_float2(0.f);
                    texcoord3d = p0;
                    rtReportIntersection(0);
                    break;
                }
            t += step;
        }
    }
}

RT_PROGRAM void intersect(int primIdx)
{
    intersect_volume<false>(primIdx);
}

RT_PROGRAM void robust_intersect(int primIdx)
{
    intersect_volume<true>(primIdx);
}

RT_PROGRAM void bounds(int primIdx, float result[6])
{
    const int idx = primIdx * volume_size;
    const float3 dimensions = {volumes[idx + OFFSET_DIMENSIONS], volumes[idx + OFFSET_DIMENSIONS + 1],
                               volumes[idx + OFFSET_DIMENSIONS + 2]};
    const float3 position = {volumes[idx + OFFSET_POSITION], volumes[idx + OFFSET_POSITION + 1],
                             volumes[idx + OFFSET_POSITION + 2]};
    const float3 spacing = {volumes[idx + OFFSET_SPACING], volumes[idx + OFFSET_SPACING + 1],
                            volumes[idx + OFFSET_SPACING + 2]};

    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = position;
    aabb->m_max = position + dimensions * spacing;
}
