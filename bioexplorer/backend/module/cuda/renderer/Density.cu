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

#include <optix_world.h>

#include <platform/engines/optix6/cuda/Context.cuh>
#include <platform/engines/optix6/cuda/Random.cuh>

// Rendering attributes
rtDeclareVariable(int, samplesPerFrame, , );
rtDeclareVariable(float, rayLength, , );
rtDeclareVariable(float, rayStep, , );
rtDeclareVariable(float, farPlane, , );

static __device__ inline void shade()
{
    optix::size_t2 screen = output_buffer.size();
    float t = rayStep;
    float4 color = make_float4(0.f);

    while (color.w < 0.9f && t < farPlane)
    {
        uint hits = 0;
        uint seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

        const float3 hit_point = ray.origin + t_hit * ray.direction;
        const float3 normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

        for (int i = 0; i < samplesPerFrame; ++i)
        {
            float attenuation = 0.f;
            float3 aa_normal = optix::normalize(make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));
            if (::optix::dot(aa_normal, normal) < 0.f)
                aa_normal = -aa_normal;

            PerRayData_shadow shadow_prd;
            shadow_prd.attenuation = make_float3(0.f);
            ::optix::Ray shadow_ray(hit_point, aa_normal, shadowRayType, sceneEpsilon, rayLength);
            rtTrace(top_shadower, shadow_ray, shadow_prd);

            attenuation += ::optix::luminance(shadow_prd.attenuation);
            if (attenuation > 0.f)
                ++hits;
        }

        if (hits > 0)
        {
            const float a = (float)hits / (float)samplesPerFrame;
            const float3 sampleColor = make_float3(a, a, 1.f - a);
            const float alpha = 1.f / (float)samplesPerFrame;
            color = make_float4(make_float3(color) * color.w + (1.f - color.w * alpha) * sampleColor, color.w + alpha);
        }
        t += rayStep;
    }
    prd.result = color;
}

RT_PROGRAM void any_hit_shadow()
{
    prd_shadow.attenuation = make_float3(1.f);
    rtTerminateRay();
}

RT_PROGRAM void closest_hit_radiance()
{
    shade();
}

RT_PROGRAM void closest_hit_radiance_textured()
{
    shade();
}
