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
rtDeclareVariable(uint, showVectorDirections, , );

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
#define DATA_SIZE 6

template <int depth>
__device__ float3 treeWalker(const uint startIndices, const uint startData, const float3& point, const float distance,
                             const float cutoff, const uint index, const float3 offset, const float3 spacing)
{
    if (depth >= MAX_RECURSION_DEPTH)
        return make_float3(0.f);

    const uint begin = userDataBuffer[startIndices + index * 2];
    const uint end = userDataBuffer[startIndices + index * 2 + 1];
    const uint idxData = startData + index * DATA_SIZE;

    if (idxData >= userDataBuffer.size())
        return make_float3(0.f);

    if (begin == 0 && end == 0)
    {
        // Leaf
        const float3 vectorDirection =
            make_float3(userDataBuffer[idxData + 3], userDataBuffer[idxData + 4], userDataBuffer[idxData + 5]);
        return vectorDirection / (distance * distance);
    }

    float3 voxelValue = make_float3(0.f);
    for (uint childIndex = begin; childIndex <= end; ++childIndex)
    {
        const uint idx = startData + childIndex * DATA_SIZE;
        const float3 childPosition = make_float3(userDataBuffer[idx], userDataBuffer[idx + 1], userDataBuffer[idx + 2]);
        const float d = length(point - childPosition);

        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate events in the child node, we take the
            // precomputed value of node instead
            const float3 vectorDirection =
                make_float3(userDataBuffer[idx + 3], userDataBuffer[idx + 4], userDataBuffer[idx + 5]);
            voxelValue += vectorDirection / (d * d);
        }
        else
            // Dive into the child node and compute its contents
            voxelValue +=
                treeWalker<depth + 1>(startIndices, startData, point, d, cutoff / 2.f, childIndex, offset, spacing);
    }
    return voxelValue;
}

template <>
__device__ float3 treeWalker<MAX_RECURSION_DEPTH>(const uint startIndices, const uint startData, const float3& point,
                                                  const float distance, const float cutoff, const uint index,
                                                  const float3 offset, const float3 spacing)
{
    return make_float3(0.f);
}

static __device__ inline void shade()
{
    float4 finalColor = make_float4(0.f);

    const float3 offset = make_float3(userDataBuffer[0], userDataBuffer[1], userDataBuffer[2]);
    const float3 spacing = make_float3(userDataBuffer[3], userDataBuffer[4], userDataBuffer[5]);
    const float3 dimensions = make_float3(userDataBuffer[6], userDataBuffer[7], userDataBuffer[8]);
    const float distance = userDataBuffer[9] * 5.f;
    const uint startIndices = 11;
    const uint startData = startIndices + userDataBuffer[10];
    const float diag = fmax(fmax(dimensions.x, dimensions.y), dimensions.z);
    const float t_step = fmax(minRayStep, diag / (float)nbRaySteps);

    float t0, t1;
    if (!intersection(offset, dimensions, spacing, ray, t0, t1))
    {
        prd.result = finalColor;
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

        const float3 sampleValue = treeWalker<0>(startIndices, startData, point, distance, cutoff, 0, offset, spacing);
        const float vectorLength = length(sampleValue);
        const float4 mapColor = calcTransferFunctionColor(transfer_function_map, value_range, vectorLength);
        if (mapColor.w > 0.f)
            if (showVectorDirections)
            {
                const float3 v = normalize(sampleValue);
                const float3 vectorColor = make_float3(0.5f + v.x * 0.5f, 0.5f + v.y * 0.5f, 0.5f + v.z * 0.5f);
                compose(make_float4(vectorColor, mapColor.w), finalColor, alphaCorrection);
            }
            else
                compose(mapColor, finalColor, alphaCorrection);

        t += t_step;
    }

    // Main exposure
    finalColor = make_float4(::optix::clamp(make_float3(finalColor * mainExposure), 0.f, 1.f), finalColor.w);

    // Environment
    compose(make_float4(getEnvironmentColor(ray.direction), 1.f), finalColor);

    prd.result = finalColor;
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
