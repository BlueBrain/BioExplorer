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

#include <platform/engines/optix6/cuda/Context.cuh>
#include <platform/engines/optix6/cuda/Environment.cuh>
#include <platform/engines/optix6/cuda/Random.cuh>
#include <platform/engines/optix6/cuda/renderer/Volume.cuh>

// Renderer
rtDeclareVariable(float, cutoff, , );
rtDeclareVariable(float, minRayStep, , );
rtDeclareVariable(int, nbRaySteps, , );
rtDeclareVariable(float, alphaCorrection, , );

static __device__ inline bool intersection(const float3& volumeOffset, const float3& volumeDimensions,
                                           const float3& volumeElementSpacing, const optix::Ray& ray, float& t0,
                                           float& t1)
{
    const float3 boxmin = volumeOffset;
    const float3 boxmax = volumeOffset + volumeDimensions / volumeElementSpacing;

    const float3 a = (boxmin - ray.origin) / ray.direction;
    const float3 b = (boxmax - ray.origin) / ray.direction;
    const float3 near = fminf(a, b);
    const float3 far = fmaxf(a, b);
    t0 = fmaxf(near);
    t1 = fminf(far);

    return (t0 <= t1);
}

/*
A smart way to avoid recursion restrictions with OptiX 6 is to use templates!

https://www.thanassis.space/cudarenderer-BVH.html#recursion
*/

#define MAX_RECURSION_DEPTH 10

template <int depth>
__device__ float treeWalker(const uint startIndices, const uint startData, const float3& point, const float distance,
                            const float cutoff, const uint index)
{
    if (depth >= MAX_RECURSION_DEPTH)
        return 0.f;

    const uint begin = simulation_data[startIndices + index * 2];
    const uint end = simulation_data[startIndices + index * 2 + 1];
    const uint idxData = startData + index * 4;

    if (idxData >= simulation_data.size())
        return 0.f;

    if (begin == 0 && end == 0)
        // Leaf
        return simulation_data[idxData + 3] / (distance * distance);

    float voxelValue = 0.f;
    for (uint childIndex = begin; childIndex <= end; ++childIndex)
    {
        const uint idx = startData + childIndex * 4;
        const float3 childPosition =
            make_float3(simulation_data[idx], simulation_data[idx + 1], simulation_data[idx + 2]);
        const float3 delta = point - childPosition;

        const float d = sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);

        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate
            // events in the child node, we take the precomputed value of node
            // instead
            voxelValue += simulation_data[idx + 3] / (d * d);
        }
        else
            // Dive into the child node and compute its contents
            voxelValue += treeWalker<depth + 1>(startIndices, startData, point, d, cutoff / 2.f, childIndex);
    }
    return voxelValue;
}

template <>
__device__ float treeWalker<MAX_RECURSION_DEPTH>(const uint startIndices, const uint startData, const float3& point,
                                                 const float distance, const float cutoff, const uint index)
{
    return 0.f;
}

static __device__ inline void shade()
{
    float4 finalColor = make_float4(0.f);

    const float3 offset = make_float3(simulation_data[0], simulation_data[1], simulation_data[2]);
    const float3 spacing = make_float3(simulation_data[3], simulation_data[4], simulation_data[5]);
    const float3 dimensions = make_float3(simulation_data[6], simulation_data[7], simulation_data[8]);
    const float distance = simulation_data[9] * 5.f;
    const uint startIndices = 11;
    const uint startData = startIndices + simulation_data[10];
    const float diag = fmax(fmax(dimensions.x, dimensions.y), dimensions.z);
    const float t_step = fmax(minRayStep, diag / (float)nbRaySteps);

    float t0, t1;
    if (!intersection(offset, dimensions, spacing, ray, t0, t1))
    {
        prd.result = make_float3(finalColor);
        return;
    }

    optix::size_t2 screen = output_buffer.size();
    uint seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);
    const float random = rnd(seed) * t_step;

    float t = fmax(0.f, t0) + random;
    while (t < t1 && finalColor.w < 1.f)
    {
        const float3 p = ray.origin + t * ray.direction;
        const float3 point = (p - offset) / spacing;

        const float sampleValue = treeWalker<0>(startIndices, startData, point, distance, cutoff, 0);
        const float4 sampleColor = calcTransferFunctionColor(transfer_function_map, value_range, sampleValue);
        if (sampleColor.w > 0.f)
            compose(sampleColor, finalColor, alphaCorrection);

        t += t_step;
    }

    // Main exposure
    finalColor = make_float4(::optix::clamp(make_float3(finalColor * mainExposure), 0.f, 1.f), finalColor.w);

    // Environment
    compose(make_float4(getEnvironmentColor(ray.direction), 1.f), finalColor);

    prd.result = make_float3(finalColor);
    prd.importance = finalColor.w;
}

RT_PROGRAM void any_hit_shadow()
{
    rtTerminateRay();
}

RT_PROGRAM void closest_hit_radiance_textured()
{
    shade();
}

RT_PROGRAM void closest_hit_radiance()
{
    shade();
}
