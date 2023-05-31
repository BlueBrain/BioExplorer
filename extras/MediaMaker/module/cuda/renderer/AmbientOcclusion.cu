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

#include <optix_world.h>

#include <core/engines/optix6/OptiXCommonStructs.h>
#include <core/engines/optix6/cuda/Random.cuh>

// Scene
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned int, frame, , );
rtDeclareVariable(unsigned int, shadowRayType, , );

// Material attributes
rtDeclareVariable(float3, Ko, , );

// Rendering attributes
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(PerRayData_radiance, prd, rtPayload, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(int, samplesPerFrame, , );
rtDeclareVariable(float, rayLength, , );
rtDeclareVariable(float, sceneEpsilon, , );

rtDeclareVariable(rtObject, top_shadower, , );

rtBuffer<uchar4, 2> output_buffer;

static __device__ inline void shade()
{
    optix::size_t2 screen = output_buffer.size();
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    const float3 hit_point = ray.origin + t_hit * ray.direction;
    const float3 normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

    float attenuation = 0.f;
    for (int i = 0; i < samplesPerFrame; ++i)
    {
        float3 aa_normal = optix::normalize(make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));
        if (::optix::dot(aa_normal, normal) < 0.f)
            aa_normal = -aa_normal;

        PerRayData_shadow shadow_prd;
        shadow_prd.attenuation = make_float3(1.f);
        ::optix::Ray shadow_ray(hit_point, aa_normal, shadowRayType, sceneEpsilon, rayLength);
        rtTrace(top_shadower, shadow_ray, shadow_prd);

        attenuation += ::optix::luminance(shadow_prd.attenuation);
    }
    attenuation = ::optix::clamp(attenuation / float(samplesPerFrame), 0.f, 1.f);
    prd.result = make_float3(attenuation);
}

RT_PROGRAM void any_hit_shadow()
{
    prd_shadow.attenuation = 1.f - Ko;
    rtTerminateRay();
}

RT_PROGRAM void closest_hit_radiance()
{
    shade();
}
