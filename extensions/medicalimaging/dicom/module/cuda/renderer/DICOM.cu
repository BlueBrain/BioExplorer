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

#include <platform/engines/optix6/cuda/Environment.cuh>
#include <platform/engines/optix6/cuda/Helpers.cuh>
#include <platform/engines/optix6/cuda/Random.cuh>
#include <platform/engines/optix6/cuda/renderer/Volume.cuh>

#include <platform/core/common/CommonTypes.h>

using namespace optix;

rtDeclareVariable(float, surfaceOffset, , );

static __device__ void dicomShade()
{
    float3 result = make_float3(0.f);

    float4 voxelColor = make_float4(Kd, luminance(Ko));
    if (volume_map != 0)
    {
        const float voxelValue = optix::rtTex3D<float>(volume_map, texcoord3d.x, texcoord3d.y, texcoord3d.z);
        voxelColor = calcTransferFunctionColor(tfMinValue, tfMinValue + tfRange, voxelValue, tfColors, tfOpacities);
    }

    const float3 hit_point = ray.origin + t_hit * ray.direction;
    float cosNL = 1.f;

    float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    optix::size_t2 screen = output_buffer.size();
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    // Shadows
    float light_attenuation = 1.f;
    if (shadows > 0.f)
    {
        unsigned int num_lights = lights.size();
        for (int i = 0; i < num_lights; ++i)
        {
            BasicLight light = lights[i];
            if (light.casts_shadow)
            {
                float3 lightDirection;

                if (light.type == BASIC_LIGHT_TYPE_POINT)
                {
                    float3 pos = light.pos;
                    if (softShadows > 0.f)
                        pos += softShadows * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                    lightDirection = normalize(pos - hit_point);
                }
                else
                {
                    lightDirection = -light.pos;
                    if (softShadows > 0.f)
                        lightDirection +=
                            softShadows * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                    lightDirection = normalize(lightDirection);
                }

                PerRayData_shadow shadow_prd;
                shadow_prd.attenuation = make_float3(1.f);
                float near = sceneEpsilon + surfaceOffset;
                float far = giDistance;
                applyClippingPlanes(hit_point, lightDirection, near, far);
                optix::Ray shadow_ray(hit_point, lightDirection, shadowRayType, near, far);
                rtTrace(top_shadower, shadow_ray, shadow_prd);

                light_attenuation -= shadows * (1.f - luminance(shadow_prd.attenuation));
            }
        }
    }

    // Shading
    if (light_attenuation > 0.f && volumeGradientShadingEnabled)
    {
        unsigned int num_lights = lights.size();
        for (int i = 0; i < num_lights; ++i)
        {
            // Phong
            BasicLight light = lights[i];
            float3 lightDirection;

            if (light.type == BASIC_LIGHT_TYPE_POINT)
            {
                float3 pos = light.pos;
                if (shadows > 0.f && softShadows > 0.f)
                    // Soft shadows
                    pos += softShadows * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                lightDirection = normalize(pos - hit_point);
            }
            else
            {
                lightDirection = -light.pos;
                if (shadows > 0.f && softShadows > 0.f)
                    // Soft shadows
                    lightDirection += softShadows * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                lightDirection = normalize(lightDirection);
            }
            cosNL = max(0.f, dot(normal, lightDirection));
            cosNL = DEFAULT_SHADING_AMBIENT + (1.f - DEFAULT_SHADING_AMBIENT) * cosNL;
            float3 specularColor = make_float3(0.f);

            // Specular
            const float power = pow(cosNL, specularExponent);
            specularColor = power * volumeSpecularColor;

            voxelColor = make_float4(make_float3(voxelColor) * cosNL + specularColor, voxelColor.w);
        }

        // Alpha ratio
        voxelColor = make_float4(make_float3(voxelColor) * DEFAULT_SHADING_ALPHA_RATIO, voxelColor.w);
    }
    result = make_float3(voxelColor) * light_attenuation * voxelColor.w;

    // Refraction
    const float opacity = voxelColor.w;
    if (voxelColor.w < 1.f && prd.depth < maxBounces)
    {
        PerRayData_radiance refracted_prd;
        refracted_prd.result = make_float3(0.f);
        refracted_prd.importance = prd.importance * (1.f - opacity);
        refracted_prd.depth = prd.depth + 1;

        const optix::Ray refracted_ray(hit_point, ray.direction, radianceRayType, sceneEpsilon);
        rtTrace(top_object, refracted_ray, refracted_prd);
        result = result * opacity + (1.f - opacity) * refracted_prd.result;
    }

    // Fog attenuation
    const float z = optix::length(eye - hit_point);
    const float fogAttenuation = z > fogStart ? optix::clamp((z - fogStart) / fogThickness, 0.f, 1.f) : 0.f;
    result = (result * (1.f - fogAttenuation) + fogAttenuation * getEnvironmentColor(ray.direction));

    // Final result
    prd.result = result;
}

RT_PROGRAM void any_hit_shadow()
{
    prd_shadow.attenuation = 1.f - Ko;
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
    output_buffer[launch_index] = make_color(make_float3(0, 1, 0));
}
