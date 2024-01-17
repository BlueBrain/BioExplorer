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

#include <platform/engines/optix6/cuda/Helpers.cuh>
#include <platform/engines/optix6/cuda/Random.cuh>

rtDeclareVariable(int, shadowSamples, , );

static __device__ inline void shade()
{
    optix::size_t2 screen = output_buffer.size();
    uint seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    const float3 hit_point = ray.origin + t_hit * ray.direction;
    const float3 normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));

    uint num_lights = lights.size();
    float attenuation = 0.f;
    for (int s = 0; s < shadowSamples; ++s)
    {
        for (int i = 0; i < num_lights; ++i)
        {
            BasicLight light = lights[i];
            float3 lightDirection;
            if (light.type == BASIC_LIGHT_TYPE_POINT)
            {
                // Point light
                float3 pos = light.pos;
                if (softShadowStrength > 0.f)
                    // Soft shadowIntensity
                    pos += softShadowStrength * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                lightDirection = optix::normalize(pos - hit_point);
            }
            else
            {
                // Directional light
                lightDirection = -light.pos;
                if (softShadowStrength > 0.f)
                    // Soft shadowIntensity
                    lightDirection +=
                        softShadowStrength * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                lightDirection = optix::normalize(lightDirection);
            }
            float nDl = optix::dot(normal, lightDirection);

            // Shadows
            if (nDl > 0.f && light.casts_shadow)
            {
                PerRayData_shadow shadow_prd;
                shadow_prd.attenuation = make_float3(1.f);
                float near = sceneEpsilon;
                float far = giRayLength;
                applyClippingPlanes(hit_point, lightDirection, near, far);
                optix::Ray shadow_ray(hit_point, lightDirection, shadowRayType, near, far);
                rtTrace(top_shadower, shadow_ray, shadow_prd);

                // light_attenuation is zero if completely shadowed
                attenuation += ::optix::luminance(shadow_prd.attenuation);
            }
        }
    }
    attenuation = ::optix::clamp(attenuation / float(shadowSamples), 0.f, 1.f);
    prd.result = make_float4(make_float3(attenuation), 1.f);
    prd.zDepth = optix::length(eye - hit_point);
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

RT_PROGRAM void closest_hit_radiance_textured()
{
    shade();
}
