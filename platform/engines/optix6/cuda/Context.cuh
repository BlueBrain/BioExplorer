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

#pragma once

#include <platform/engines/optix6/OptiXCommonStructs.h>

#include <optixu/optixu_math_namespace.h>

// Scene
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(unsigned int, radianceRayType, , );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(unsigned int, shadowRayType, , );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );

rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, dir, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(float, sceneEpsilon, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(unsigned int, radiance_ray_type, , );
rtDeclareVariable(unsigned int, frame, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );

rtBuffer<uchar4, 2> output_buffer;
rtBuffer<float4, 2> accum_buffer;

rtDeclareVariable(float, height, , );
rtDeclareVariable(float4, jitter4, , );
rtDeclareVariable(unsigned int, samples_per_pixel, , );

// Material attributes
rtDeclareVariable(float3, Ka, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float3, Ks, , );
rtDeclareVariable(float3, Kr, , );
rtDeclareVariable(float3, Ko, , );
rtDeclareVariable(float, glossiness, , );
rtDeclareVariable(float, refraction_index, , );
rtDeclareVariable(float, phong_exp, , );
rtDeclareVariable(uint, shading_mode, , );
rtDeclareVariable(float, user_parameter, , );
rtDeclareVariable(uint, cast_user_data, , );
rtDeclareVariable(uint, clipping_mode, , );

// Textures
rtDeclareVariable(int, albedoMetallic_map, , );
rtDeclareVariable(float2, texcoord, attribute texcoord, );

// Shading
rtDeclareVariable(float3, bgColor, , );
rtDeclareVariable(int, envmap, , );
rtDeclareVariable(uint, use_envmap, , );
rtDeclareVariable(uint, showBackground, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(float, mainExposure, , );

// Clipping planes
rtBuffer<float4, 1> clippingPlanes;
rtDeclareVariable(unsigned int, nbClippingPlanes, , );
rtDeclareVariable(unsigned int, enableClippingPlanes, , );

// Camera
rtDeclareVariable(float, apertureRadius, , );
rtDeclareVariable(float, focusDistance, , );
rtDeclareVariable(unsigned int, stereo, , );
rtDeclareVariable(float3, ipd_offset, , );

// Lights
rtBuffer<BasicLight> lights;
rtDeclareVariable(float3, ambientLightColor, , );

// Transfer function
rtBuffer<float3> tfColors;
rtBuffer<float> tfOpacities;
rtDeclareVariable(float, tfMinValue, , );
rtDeclareVariable(float, tfRange, , );
rtDeclareVariable(uint, tfSize, , );

// Volume
rtDeclareVariable(uint3, volumeDimensions, , );
rtDeclareVariable(float3, volumeOffset, , );
rtDeclareVariable(float3, volumeElementSpacing, , );
rtDeclareVariable(uint, volumeSamplesPerRay, , );
rtDeclareVariable(uint, volumeDataTypeSize, , );
rtDeclareVariable(uint, volumeDataType, , );
rtDeclareVariable(int, volumeSampler, , );
rtBuffer<float3> volumeGradientBuffer;

// Volume shading
rtDeclareVariable(uint, volumeGradientShadingEnabled, , );
rtDeclareVariable(float, volumeAdaptiveMaxSamplingRate, , );
rtDeclareVariable(uint, volumeSingleShade, , );
rtDeclareVariable(float, volumeSamplingRate, , );
rtDeclareVariable(float3, volumeSpecular, , );
rtDeclareVariable(float, volumeNormalEpsilon, , );

// Simulation data
rtBuffer<float> simulation_data;
rtDeclareVariable(unsigned long, simulation_idx, attribute simulation_idx, );
