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

static __device__ inline void shade()
{
    optix::size_t2 screen = output_buffer.size();
    uint seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    const float3 hit_point = ray.origin + t_hit * ray.direction;
    const float3 world_shading_normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    const float3 world_geometric_normal = optix::normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    float3 normal = optix::faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

    // Glossiness
    if (glossiness < 1.f)
        normal = optix::normalize(normal + (1.f - glossiness) *
                                               make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));

    // Diffuse
    float attenuation = 0.f;
    for (int i = 0; i < giSamples; ++i)
    {
        float3 aa_normal = optix::normalize(make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));
        if (::optix::dot(aa_normal, normal) < 0.f)
            aa_normal = -aa_normal;

        PerRayData_shadow shadow_prd;
        shadow_prd.attenuation = make_float3(1.f);

        float near = sceneEpsilon;
        float far = giRayLength;
        applyClippingPlanes(hit_point, aa_normal, near, far);

        ::optix::Ray shadow_ray(hit_point, aa_normal, shadowRayType, near, far);
        rtTrace(top_shadower, shadow_ray, shadow_prd);

        attenuation += ::optix::luminance(shadow_prd.attenuation);
    }
    float3 color = make_float3((attenuation) / float(giSamples));

    const float opacity = fmaxf(Ko);
    if (opacity > 0.f && prd.depth < maxRayDepth - 1)
    {
        const float reflection = fmaxf(Kr);
        // Reflection
        if (reflection > 0.f)
        {
            PerRayData_radiance reflected_prd;
            reflected_prd.result = make_float4(0.f);
            reflected_prd.importance = prd.importance * reflection;
            reflected_prd.depth = prd.depth + 1;

            const float3 R = optix::reflect(ray.direction, normal);
            const optix::Ray reflected_ray(hit_point, R, radianceRayType, sceneEpsilon, giRayLength);
            rtTrace(top_object, reflected_ray, reflected_prd);
            color = color * (1.f - reflection) + Kr * make_float3(reflected_prd.result);
        }

        // Refraction
        if (opacity < 1.f)
        {
            PerRayData_radiance refracted_prd;
            refracted_prd.result = make_float4(0.f);
            refracted_prd.importance = prd.importance * (1.f - opacity);
            refracted_prd.depth = prd.depth + 1;

            const float3 refractedNormal = refractedVector(ray.direction, normal, refraction_index, 1.f);
            const optix::Ray refracted_ray(hit_point, refractedNormal, radianceRayType, sceneEpsilon, giRayLength);
            rtTrace(top_object, refracted_ray, refracted_prd);
            color = color * opacity + (1.f - opacity) * make_float3(refracted_prd.result);
        }
    }

    prd.result = make_float4(::optix::clamp(color, 0.f, 1.f), opacity);
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
