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

#include <science/common/CommonTypes.h>

#include <platform/engines/optix6/cuda/Context.cuh>
#include <platform/engines/optix6/cuda/Environment.cuh>
#include <platform/engines/optix6/cuda/Random.cuh>
#include <platform/engines/optix6/cuda/renderer/Volume.cuh>

#define MAX_RECURSION_DEPTH 15

// Renderer
rtDeclareVariable(float, cutoff, , );
rtDeclareVariable(float, minRayStep, , );
rtDeclareVariable(int, nbRaySteps, , );
rtDeclareVariable(int, nbRayRefinementSteps, , );
rtDeclareVariable(float, alphaCorrection, , );

/**
A smart way to avoid recursion restrictions with OptiX 6 is to use templates!

https://www.thanassis.space/cudarenderer-BVH.html#recursion
*/
template <int depth>
__device__ float treeWalker(const uint startIndices, const uint startData, const float3& point, const float distance,
                            const float cutoff, const uint index)
{
    if (depth >= MAX_RECURSION_DEPTH)
        return 0.f;

    const uint begin = userDataBuffer[startIndices + index * 2];
    const uint end = userDataBuffer[startIndices + index * 2 + 1];
    const uint idxData = startData + index * FIELD_POINT_DATA_SIZE;

    if (idxData >= userDataBuffer.size())
        return 0.f;

    if (begin == 0 && end == 0)
        // Leaf
        return userDataBuffer[idxData + FIELD_POINT_OFFSET_VALUE] / (distance * distance);

    float voxelValue = 0.f;
    for (uint childIndex = begin; childIndex <= end; ++childIndex)
    {
        const uint idx = startData + childIndex * FIELD_POINT_DATA_SIZE;
        const float3 childPosition = make_float3(userDataBuffer[idx + FIELD_POINT_OFFSET_POSITION_X],
                                                 userDataBuffer[idx + FIELD_POINT_OFFSET_POSITION_Y],
                                                 userDataBuffer[idx + FIELD_POINT_OFFSET_POSITION_Z]);
        const float d = length(point - childPosition);
        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate events in the child node, we take the
            // precomputed value of node instead
            voxelValue += userDataBuffer[idx + FIELD_POINT_OFFSET_VALUE] / (d * d);
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

    const float3 offset = make_float3(userDataBuffer[OCTREE_DATA_OFFSET_X], userDataBuffer[OCTREE_DATA_OFFSET_Y],
                                      userDataBuffer[OCTREE_DATA_OFFSET_Z]);
    const float3 spacing = make_float3(userDataBuffer[OCTREE_DATA_SPACING_X], userDataBuffer[OCTREE_DATA_SPACING_Y],
                                       userDataBuffer[OCTREE_DATA_SPACING_Z]);
    const float3 dimensions =
        make_float3(userDataBuffer[OCTREE_DATA_DIMENSION_X], userDataBuffer[OCTREE_DATA_DIMENSION_Y],
                    userDataBuffer[OCTREE_DATA_DIMENSION_Z]);
    const float distance = userDataBuffer[OCTREE_DATA_INITIAL_DISTANCE] * 5.f;
    const uint startIndices = OCTREE_DATA_INDICES;
    const uint startData = startIndices + userDataBuffer[OCTREE_DATA_VALUES];
    const float diag = fmax(fmax(dimensions.x, dimensions.y), dimensions.z);
    const float t_step = fmax(minRayStep, diag / (float)nbRaySteps);

    float t0, t1;
    if (!boxIntersection(offset, dimensions, spacing, ray, t0, t1))
    {
        prd.result = finalColor;
        return;
    }

    optix::size_t2 screen = output_buffer.size();
    uint seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);
    const float random = (frame > 0 ? rnd(seed) * t_step : 0.f);

    float t = fmax(0.f, t0) + random;
    while (t < t1 && finalColor.w < 1.f)
    {
        const float3 p = ray.origin + t * ray.direction;
        const float3 point = (p - offset) / spacing;

        const float value = treeWalker<0>(startIndices, startData, point, distance, cutoff, 0);
        const float4 sampleColor = calcTransferFunctionColor(transfer_function_map, value_range, value);
        if (sampleColor.w > 0.f)
            compose(sampleColor, finalColor, alphaCorrection);

        t += t_step;
    }

    // Main exposure
    finalColor = make_float4(::optix::clamp(make_float3(finalColor * mainExposure), 0.f, 1.f), finalColor.w);

    // Environment
    compose(make_float4(getEnvironmentColor(ray.direction), 1.f), finalColor, alphaCorrection);

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
