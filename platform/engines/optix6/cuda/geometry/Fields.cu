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

#include <platform/core/common/CommonTypes.h>
#include <platform/engines/optix6/cuda/Random.cuh>
#include <platform/engines/optix6/cuda/renderer/Volume.cuh>

using namespace optix;

const uint FIELD_OFFSET_DIMENSIONS = 0;
const uint FIELD_OFFSET_OFFSET = FIELD_OFFSET_DIMENSIONS + 3;
const uint FIELD_OFFSET_SPACING = FIELD_OFFSET_OFFSET + 3;
const uint FIELD_OFFSET_TRANSFER_FUNCTION_TEXTURE_SAMPLER_ID = FIELD_OFFSET_SPACING + 3;
const uint FIELD_OFFSET_VALUE_RANGE = FIELD_OFFSET_TRANSFER_FUNCTION_TEXTURE_SAMPLER_ID + 1;
const uint FIELD_OFFSET_OCTREE_INDICES_SAMPLER_ID = FIELD_OFFSET_VALUE_RANGE + 2;
const uint FIELD_OFFSET_OCTREE_VALUES_SAMPLER_ID = FIELD_OFFSET_OCTREE_INDICES_SAMPLER_ID + 1;
const uint FIELD_OFFSET_OCTREE_TYPE = FIELD_OFFSET_OCTREE_VALUES_SAMPLER_ID + 1;
const uint FIELD_OFFSET_NB_VALUES = FIELD_OFFSET_OCTREE_TYPE + 1;

rtDeclareVariable(uint, field_size, , );

rtBuffer<float> fields;

/**
A smart way to avoid recursion restrictions with OptiX 6 is to use templates!

https://www.thanassis.space/cudarenderer-BVH.html#recursion
*/
#define MAX_RECURSION_DEPTH 30

template <int depth>
__device__ float treeWalker(const int fieldOctreeIndicesId, const int fieldOctreeValuesId, const float3& point,
                            const float distance, const float cutoff, const uint index)
{
    if (depth >= MAX_RECURSION_DEPTH)
        return 0.f;

    const uint begin = optix::rtTex1D<uint32_t>(fieldOctreeIndicesId, index * 2);
    const uint end = optix::rtTex1D<uint32_t>(fieldOctreeIndicesId, index * 2 + 1);
    const uint idxData = index * FIELD_POINT_DATA_SIZE;

    if (begin == 0 && end == 0)
    {
        // Leaf
        const float value = optix::rtTex1D<float>(fieldOctreeValuesId, idxData + FIELD_POINT_OFFSET_VALUE);
        return value / (distance * distance);
    }

    float voxelValue = 0.f;
    for (uint childIndex = begin; childIndex <= end; ++childIndex)
    {
        const uint childIdxData = childIndex * FIELD_POINT_DATA_SIZE;
        const float3 childPosition =
            make_float3(optix::rtTex1D<float>(fieldOctreeValuesId, childIdxData + FIELD_POINT_OFFSET_POSITION_X),
                        optix::rtTex1D<float>(fieldOctreeValuesId, childIdxData + FIELD_POINT_OFFSET_POSITION_Y),
                        optix::rtTex1D<float>(fieldOctreeValuesId, childIdxData + FIELD_POINT_OFFSET_POSITION_Z));
        const float d = length(point - childPosition);

        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate events in the child node, we take the
            // precomputed value of node instead
            const float value = optix::rtTex1D<float>(fieldOctreeValuesId, childIdxData + FIELD_POINT_OFFSET_VALUE);
            voxelValue += value / (d * d);
        }
        else
            // Dive into the child node and compute its contents
            voxelValue +=
                treeWalker<depth + 1>(fieldOctreeIndicesId, fieldOctreeValuesId, point, d, cutoff / 2.f, childIndex);
    }
    return voxelValue;
}

template <>
__device__ float treeWalker<MAX_RECURSION_DEPTH>(const int fieldOctreeIndicesId, const int fieldOctreeValuesId,
                                                 const float3& point, const float distance, const float cutoff,
                                                 const uint index)
{
    return 0.f;
}

template <int depth>
__device__ float3 treeWalker3(const int fieldOctreeIndicesId, const int fieldOctreeValuesId, const float3& point,
                              const float distance, const float cutoff, const uint index)
{
    if (depth >= MAX_RECURSION_DEPTH)
        return make_float3(0.f);

    const uint begin = optix::rtTex1D<uint32_t>(fieldOctreeIndicesId, index * 2);
    const uint end = optix::rtTex1D<uint32_t>(fieldOctreeIndicesId, index * 2 + 1);
    const uint idxData = index * FIELD_VECTOR_DATA_SIZE;

    if (begin == 0 && end == 0)
    {
        // Leaf
        const float3 vectorDirection =
            make_float3(optix::rtTex1D<float>(fieldOctreeValuesId, idxData + FIELD_VECTOR_OFFSET_DIRECTION_X),
                        optix::rtTex1D<float>(fieldOctreeValuesId, idxData + FIELD_VECTOR_OFFSET_DIRECTION_Y),
                        optix::rtTex1D<float>(fieldOctreeValuesId, idxData + FIELD_VECTOR_OFFSET_DIRECTION_Z));
        return vectorDirection / (distance * distance);
    }

    float3 voxelValue = make_float3(0.f);
    for (uint childIndex = begin; childIndex <= end; ++childIndex)
    {
        const uint idx = childIndex * FIELD_VECTOR_DATA_SIZE;
        const float3 childPosition =
            make_float3(optix::rtTex1D<float>(fieldOctreeValuesId, idx + FIELD_VECTOR_OFFSET_POSITION_X),
                        optix::rtTex1D<float>(fieldOctreeValuesId, idx + FIELD_VECTOR_OFFSET_POSITION_Y),
                        optix::rtTex1D<float>(fieldOctreeValuesId, idx + FIELD_VECTOR_OFFSET_POSITION_Z));
        const float d = length(point - childPosition);

        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate events in the child node, we take the
            // precomputed value of node instead
            const float3 vectorDirection =
                make_float3(optix::rtTex1D<float>(fieldOctreeValuesId, idx + FIELD_VECTOR_OFFSET_DIRECTION_X),
                            optix::rtTex1D<float>(fieldOctreeValuesId, idx + FIELD_VECTOR_OFFSET_DIRECTION_Y),
                            optix::rtTex1D<float>(fieldOctreeValuesId, idx + FIELD_VECTOR_OFFSET_DIRECTION_Z));
            voxelValue += vectorDirection / (d * d);
        }
        else
            // Dive into the child node and compute its contents
            voxelValue +=
                treeWalker3<depth + 1>(fieldOctreeIndicesId, fieldOctreeValuesId, point, d, cutoff / 2.f, childIndex);
    }
    return voxelValue;
}

template <>
__device__ float3 treeWalker3<MAX_RECURSION_DEPTH>(const int fieldOctreeIndicesId, const int fieldOctreeValuesId,
                                                   const float3& point, const float distance, const float cutoff,
                                                   const uint index)
{
    return make_float3(0.f);
}

static __device__ float get_field_value(const int fieldOctreeValuesId, const float3& point, const int nbValues)
{
    float result = 0.f;
    for (uint i = 0; i < nbValues; ++i)
    {
        const uint index = i * FIELD_POINT_DATA_SIZE;
        const float3 position =
            make_float3(optix::rtTex1D<float>(fieldOctreeValuesId, index + FIELD_POINT_OFFSET_POSITION_X),
                        optix::rtTex1D<float>(fieldOctreeValuesId, index + FIELD_POINT_OFFSET_POSITION_Y),
                        optix::rtTex1D<float>(fieldOctreeValuesId, index + FIELD_POINT_OFFSET_POSITION_Z));
        const float d = length(point - position);
        if (d < fieldCutoff)
        {
            const float value = optix::rtTex1D<float>(fieldOctreeValuesId, index + FIELD_POINT_OFFSET_VALUE);
            result += value / (d * d);
        }
    }
    return result;
}

static __device__ float4 get_voxel_value(const int idx, const float3& point, const int nbValues)
{
    const int fieldOctreeIndicesId = static_cast<int>(fields[idx + FIELD_OFFSET_OCTREE_INDICES_SAMPLER_ID]);
    const int fieldOctreeValuesId = static_cast<int>(fields[idx + FIELD_OFFSET_OCTREE_VALUES_SAMPLER_ID]);
    if (fieldOctreeIndicesId != 0 && fieldOctreeValuesId != 0)
    {
        if (fieldUseOctree)
        {
            const int fieldOctreeType = static_cast<int>(fields[idx + FIELD_OFFSET_OCTREE_TYPE]);
            switch (fieldOctreeType)
            {
            case odt_points:
                return make_float4(0.f, 0.f, 0.f,
                                   treeWalker<0>(fieldOctreeIndicesId, fieldOctreeValuesId, point, fieldDistance,
                                                 fieldCutoff, 0u));
            case odt_vectors:
                const float3 sampleValue =
                    treeWalker3<0>(fieldOctreeIndicesId, fieldOctreeValuesId, point, fieldDistance, fieldCutoff, 0u);
                return make_float4(normalize(sampleValue), length(sampleValue));
            }
        }
        else
            return make_float4(0.f, 0.f, 0.f, get_field_value(fieldOctreeValuesId, point, nbValues));
    }
    return make_float4(0.f);
}

template <bool use_robust_method>
static __device__ void intersect_field(int primIdx)
{
    const int idx = primIdx * field_size;
    const float3 dimensions = {fields[idx + FIELD_OFFSET_DIMENSIONS], fields[idx + FIELD_OFFSET_DIMENSIONS + 1],
                               fields[idx + FIELD_OFFSET_DIMENSIONS + 2]};
    const float3 offset = {fields[idx + FIELD_OFFSET_OFFSET], fields[idx + FIELD_OFFSET_OFFSET + 1],
                           fields[idx + FIELD_OFFSET_OFFSET + 2]};
    const float3 spacing = {fields[idx + FIELD_OFFSET_SPACING], fields[idx + FIELD_OFFSET_SPACING + 1],
                            fields[idx + FIELD_OFFSET_SPACING + 2]};
    const int transferFunctionSamplerId =
        static_cast<int>(fields[idx + FIELD_OFFSET_TRANSFER_FUNCTION_TEXTURE_SAMPLER_ID]);
    const float2 valueRange = {fields[idx + FIELD_OFFSET_VALUE_RANGE], fields[idx + FIELD_OFFSET_VALUE_RANGE + 1]};
    const int nbValues = fields[idx + FIELD_OFFSET_NB_VALUES];

    const float3 boxMin = offset;
    const float3 boxMax = offset + dimensions * spacing;

    const float3 a = (boxMin - ray.origin) / ray.direction;
    const float3 b = (boxMax - ray.origin) / ray.direction;
    const float3 near = fminf(a, b);
    const float3 far = fmaxf(a, b);
    float t0 = fmaxf(near);
    float t1 = fminf(far);

    const ::optix::size_t2 screen = output_buffer.size();
    uint seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    const float diag = max(spacing.x, max(spacing.y, spacing.z));
    const float step = max(0.1f, diag / fieldSamplingRate);
    const float random = fieldAccumulationSteps > 0
                             ? float(frame % fieldAccumulationSteps) / float(fieldAccumulationSteps) * step
                             : rnd(seed) * step;

    // Apply ray clipping
    t0 = max(t0, ray.tmin);
    t1 = min(t1, ray.tmax);

    if (t0 > 0.f && t0 <= t1)
    {
        float t = t0 + random;
        while (t < t1)
        {
            const float3 p = ray.origin + t * ray.direction;
            const float3 p0 = (p - offset) / spacing;
            const float4 voxelValue = get_voxel_value(idx, p0, nbValues);
            const float4 voxelColor = calcTransferFunctionColor(transferFunctionSamplerId, valueRange, voxelValue.w);
            if (voxelColor.w > 0.f)
                if (rtPotentialIntersection(t - sceneEpsilon + fieldEpsilon))
                {
                    float3 normal = make_float3(voxelValue);
                    float value = voxelValue.w;
                    if (fieldGradientShadingEnabled)
                    {
                        normal = make_float3(0);
                        const float3 positions[6] = {{-1, 0, 0}, {1, 0, 0},  {0, -1, 0},
                                                     {0, 1, 0},  {0, 0, -1}, {0, 0, 1}};
                        for (const auto& position : positions)
                        {
                            const float3 p1 = p0 + (position * fieldGradientOffset);
                            const float4 voxelValue = get_voxel_value(idx, p1, nbValues);
                            value += voxelValue.w;
                            normal += voxelValue.w * position;
                        }
                        normal = ::optix::normalize(-1.f * normal);
                        value /= 7.f;
                    }

                    geometric_normal = shading_normal = normal;
                    userDataIndex = 0;
                    texcoord = make_float2(value, 0.f);
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
    intersect_field<false>(primIdx);
}

RT_PROGRAM void robust_intersect(int primIdx)
{
    intersect_field<true>(primIdx);
}

RT_PROGRAM void bounds(int primIdx, float result[6])
{
    const int idx = primIdx * field_size;
    const float3 dimensions = {fields[idx + FIELD_OFFSET_DIMENSIONS], fields[idx + FIELD_OFFSET_DIMENSIONS + 1],
                               fields[idx + FIELD_OFFSET_DIMENSIONS + 2]};
    const float3 offset = {fields[idx + FIELD_OFFSET_OFFSET], fields[idx + FIELD_OFFSET_OFFSET + 1],
                           fields[idx + FIELD_OFFSET_OFFSET + 2]};
    const float3 spacing = {fields[idx + FIELD_OFFSET_SPACING], fields[idx + FIELD_OFFSET_SPACING + 1],
                            fields[idx + FIELD_OFFSET_SPACING + 2]};

    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = offset;
    aabb->m_max = offset + dimensions * spacing;
}
