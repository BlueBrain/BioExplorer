/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include <brayns/OptiXCommonStructs.h>

#include <brayns/cuda/Random.cuh>
#include <brayns/cuda/renderer/TransferFunction.cuh>

// Scene
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, frame, , );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

// Renderer
rtDeclareVariable(float, cutoff, , );
rtDeclareVariable(float, minRayStep, , );
rtDeclareVariable(int, nbRaySteps, , );
rtDeclareVariable(float, alphaCorrection, , );

// Simulation data
rtBuffer<float> simulation_data;
rtDeclareVariable(unsigned long, simulation_idx, attribute simulation_idx, );

// Transfer function
rtBuffer<float3> tfColors;
rtBuffer<float> tfOpacities;
rtDeclareVariable(float, tfMinValue, , );
rtDeclareVariable(float, tfRange, , );
rtDeclareVariable(uint, tfSize, , );

rtBuffer<uchar4, 2> output_buffer;

const uint STACK_SIZE = 20;

static __device__ inline bool volumeIntersection(
    const float3& volumeOffset, const float3& volumeDimensions,
    const float3& volumeElementSpacing, const optix::Ray& ray, float& t0,
    float& t1)
{
    float3 boxmin = volumeOffset + make_float3(0.f);
    float3 boxmax = volumeOffset + volumeDimensions / volumeElementSpacing;

    float3 a = (boxmin - ray.origin) / ray.direction;
    float3 b = (boxmax - ray.origin) / ray.direction;
    float3 near = fminf(a, b);
    float3 far = fmaxf(a, b);
    t0 = fmaxf(near);
    t1 = fminf(far);

    return (t0 <= t1);
}

static __device__ inline float treeWalker(
    const uint startIndices, const uint startData,
    const float3 volumeElementSpacing, const float3& point,
    const float distance, const float cutoff, const uint index = 0)
{
    return 1.f;
    float voxelValue = 0.f;

    uint nodeStack[STACK_SIZE];
    int top = 0;
    nodeStack[top] = index;
    top++;

    uint iteration = 0;
    while (top >= 0 && iteration < STACK_SIZE / 2)
    {
        const uint currentIndex = nodeStack[top];
        top--;

        const uint begin = simulation_data[startIndices + currentIndex * 2];
        const uint end = simulation_data[startIndices + currentIndex * 2 + 1];

        const uint idxData = startData + begin * 4;

        if (begin == 0 && end == 0)
            voxelValue += simulation_data[idxData + 3] / (distance * distance);
        else
        {
            uint idxLeft = begin;
            const uint idxRight = end;
            const uint idxLeftData = startData + idxLeft * 4;
            const float3 childCenter =
                make_float3(simulation_data[idxLeftData],
                            simulation_data[idxLeftData + 1],
                            simulation_data[idxLeftData + 2]);
            const float3 delta = point - childCenter;
            float d =
                sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);

            if (d < volumeElementSpacing.x / 2.f)
                d = volumeElementSpacing.x / 2.f;

            if (d < cutoff)
            {
                top++;
                nodeStack[top] = idxLeft;
                if (idxLeft != idxRight)
                {
                    top++;
                    nodeStack[top] = idxRight;
                }
            }
            else
                voxelValue += simulation_data[idxData + 3] / (d * d);
        }
        ++iteration;
    }

    return voxelValue;
}

static __device__ inline void shade()
{
    float4 finalColor = make_float4(0.f);

    const float3 offset =
        make_float3(simulation_data[0], simulation_data[1], simulation_data[2]);
    const float3 spacing =
        make_float3(simulation_data[3], simulation_data[4], simulation_data[5]);
    const float3 dimensions =
        make_float3(simulation_data[6], simulation_data[7], simulation_data[8]);
    const float distance = simulation_data[9] * 5.f; // Octree size * 5
    const uint startIndices = 11;
    const uint startData = startIndices + simulation_data[10];
    const float diag = max(max(dimensions.x, dimensions.y), dimensions.z);
    const float step = max(minRayStep, diag / nbRaySteps);

    float t0, t1;
    if (!volumeIntersection(offset, dimensions, spacing, ray, t0, t1))
    {
        prd.result = make_float3(finalColor);
        return;
    }

    optix::size_t2 screen = output_buffer.size();
    uint seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    float t = max(t_hit, t0);
    while (t < t1 && finalColor.w < 1.f)
    {
        const float3 p = ray.origin + t_hit * ray.direction;
        const float3 point = (p - offset) / spacing;

        const float value = treeWalker(startIndices, startData, spacing, point,
                                       distance, cutoff);
        const float4 sampleColor =
            calcTransferFunctionColor(tfMinValue, tfMinValue + tfRange, value,
                                      tfColors, tfOpacities);
        const float alpha = (finalColor.w == 0.0 ? 1.f : finalColor.w) *
                            alphaCorrection * sampleColor.w;
        finalColor =
            finalColor + make_float4(make_float3(sampleColor) * alpha, alpha);

        t += step;
    }

    prd.result = make_float3(finalColor);
}

RT_PROGRAM void any_hit_shadow()
{
    rtTerminateRay();
}

RT_PROGRAM void closest_hit_radiance()
{
    shade();
}
