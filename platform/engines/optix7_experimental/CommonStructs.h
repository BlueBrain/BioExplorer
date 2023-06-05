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

#pragma once

#include "cuda/geometry/GeometryData.h"

#include <optix.h>
#include <optix_stubs.h>
#include <vector_types.h>

#define BASIC_LIGHT_TYPE_POINT 0
#define BASIC_LIGHT_TYPE_DIRECTIONAL 1

namespace core
{
enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

// Light
struct BasicLight
{
    float3 pos;
    float3 color;
};

// Geometry
const unsigned int SPHERE_NUM_ATTRIBUTE_VALUES = 3u;

// Global context
struct Params
{
    unsigned int subframe_index;
    float4* accum_buffer;
    uchar4* frame_buffer;
    unsigned int width;
    unsigned int height;

    BasicLight light; // TODO: make light list
    float3 ambient_light_color;
    int max_depth;
    float scene_epsilon;

    OptixTraversableHandle handle;
};

struct State
{
    OptixDeviceContext context = 0;
    OptixTraversableHandle gas_handle = {};
    CUdeviceptr d_gas_output_buffer = {};

    OptixModule material_module = 0;
    OptixModule sphere_module = 0;
    OptixModule cylinder_module = 0;
    OptixModule cone_module = 0;
    OptixModule camera_module = 0;
    OptixModule shading_module = 0;

    OptixProgramGroup raygen_prog_group = 0;
    OptixProgramGroup miss_prog_group = 0;
    OptixProgramGroup sphere_radiance_prog_group = 0;
    OptixProgramGroup sphere_occlusion_prog_group = 0;
    OptixProgramGroup cylinder_radiance_prog_group = 0;
    OptixProgramGroup cylinder_occlusion_prog_group = 0;
    OptixProgramGroup cone_radiance_prog_group = 0;
    OptixProgramGroup cone_occlusion_prog_group = 0;

    OptixPipeline pipeline = 0;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipelineLinkOptions pipeline_link_options = {};
    OptixModuleCompileOptions module_compile_options = {};
    OptixProgramGroupOptions program_group_options = {};

    CUstream stream = 0;
    Params params;
    Params* d_params = nullptr;

    OptixShaderBindingTable sbt = {};
};

struct PerRayData_radiance
{
    float3 result;
    float importance;
    int depth;
    float3 rayDdx;
    float3 rayDdy;
};

struct RayGenData
{
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
};
typedef Record<RayGenData> RayGenRecord;

struct MissData
{
    float3 bg_color;
};
typedef Record<MissData> MissRecord;

struct Phong
{
    float3 Ka;
    float3 Kd;
    float3 Ks;
    float3 Kr;
    float phong_exp;
};

struct Glass
{
    float importance_cutoff;
    float3 cutoff_color;
    float fresnel_exponent;
    float fresnel_minimum;
    float fresnel_maximum;
    float refraction_index;
    float3 refraction_color;
    float3 reflection_color;
    float3 extinction_constant;
    float3 shadow_attenuation;
    int refraction_maxdepth;
    int reflection_maxdepth;
};

struct CheckerPhong
{
    float3 Kd1, Kd2;
    float3 Ka1, Ka2;
    float3 Ks1, Ks2;
    float3 Kr1, Kr2;
    float phong_exp1, phong_exp2;
    float2 inv_checker_size;
};

struct HitGroupData
{
    union
    {
        GeometryData::Sphere sphere;
        GeometryData::Cylinder cylinder;
        GeometryData::Cone cone;
        GeometryData::Parallelogram parallelogram;
    } geometry;

    union
    {
        Phong phong;
        Glass glass;
        CheckerPhong checker;
    } shading;
};
typedef Record<HitGroupData> HitGroupRecord;

struct RadiancePRD
{
    float3 result;
    float importance;
    int depth;
};

struct OcclusionPRD
{
    float3 attenuation;
};
} // namespace core
