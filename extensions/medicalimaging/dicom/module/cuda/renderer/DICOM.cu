/*
 * Copyright (c) 2020, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include <platform/engines/optix6/cuda/Environment.cuh>
#include <platform/engines/optix6/cuda/Helpers.cuh>
#include <platform/engines/optix6/cuda/Random.cuh>
#include <platform/engines/optix6/cuda/renderer/Volume.cuh>

#include <platform/core/common/CommonTypes.h>

using namespace optix;

static __device__ void dicomShade()
{
    // Volume contribution
    float4 volumeColor = getVolumeContribution(ray);

    // Apply exposure
    volumeColor = make_float4(make_float3(volumeColor) * mainExposure, volumeColor.w);
    float3 result = make_float3(::optix::clamp(volumeColor, 0.f, 1.f));

    // Fog attenuation
    const float3 hit_point = ray.origin + t_hit * ray.direction;
    const float z = optix::length(eye - hit_point);
    const float fogAttenuation = z > fogStart ? optix::clamp((z - fogStart) / fogThickness, 0.f, 1.f) : 0.f;
    result = (result * (1.f - fogAttenuation) + fogAttenuation * getEnvironmentColor(ray.direction));

    // Final result
    prd.result = result;
    prd.importance = volumeColor.w;
}

RT_PROGRAM void any_hit_shadow()
{
    rtTerminateRay();
}

RT_PROGRAM void closest_hit_radiance()
{
    dicomShade();
}

RT_PROGRAM void closest_hit_radiance_textured()
{
    dicomShade();
}

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_color(bad_color);
}
