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

static __device__ void dicomShade(const bool textured)
{
    float3 result = make_float3(0.f);

    float4 voxelColor = make_float4(Kd, luminance(Ko));
    if (textured)
    {
        if (volume_map != 0)
        {
            const float voxelValue = rtTex3D<float>(volume_map, texcoord3d.x, texcoord3d.y, texcoord3d.z);
            voxelColor = calcTransferFunctionColor(transfer_function_map, value_range, voxelValue);
        }
        else if (octree_indices_map != 0 && octree_values_map != 0)
            voxelColor = calcTransferFunctionColor(transfer_function_map, value_range, texcoord.x);
        else
            voxelColor = make_float4(make_float3(optix::rtTex2D<float4>(albedoMetallic_map, texcoord.x, texcoord.y)),
                                     luminance(Ko));
    }

    const float3 hit_point = ray.origin + t_hit * ray.direction;

    float3 normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    size_t2 screen = output_buffer.size();
    uint seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    // Shadows
    float light_attenuation = 1.f;
    if (shadowIntensity > 0.f)
    {
        uint num_lights = lights.size();
        for (int i = 0; i < num_lights; ++i)
        {
            BasicLight light = lights[i];
            if (light.casts_shadow)
            {
                float3 lightDirection;

                if (light.type == BASIC_LIGHT_TYPE_POINT)
                {
                    float3 pos = light.pos;
                    if (softShadowStrength > 0.f)
                        pos += softShadowStrength * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                    lightDirection = normalize(pos - hit_point);
                }
                else
                {
                    lightDirection = -light.pos;
                    if (softShadowStrength > 0.f)
                        lightDirection +=
                            softShadowStrength * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                    lightDirection = normalize(lightDirection);
                }

                PerRayData_shadow shadow_prd;
                shadow_prd.attenuation = make_float3(1.f);
                float near = sceneEpsilon + surfaceOffset;
                float far = giRayLength;
                applyClippingPlanes(hit_point, lightDirection, near, far);
                Ray shadow_ray(hit_point, lightDirection, shadowRayType, near, far);
                rtTrace(top_shadower, shadow_ray, shadow_prd);

                light_attenuation -= shadowIntensity * (1.f - luminance(shadow_prd.attenuation));
            }
        }
    }

    // Shading
    if (light_attenuation > 0.f && volumeGradientShadingEnabled)
    {
        uint num_lights = lights.size();
        for (int i = 0; i < num_lights; ++i)
        {
            // Phong
            BasicLight light = lights[i];
            float3 lightDirection;

            if (light.type == BASIC_LIGHT_TYPE_POINT)
            {
                float3 pos = light.pos;
                if (shadowIntensity > 0.f && softShadowStrength > 0.f)
                    // Soft shadowIntensity
                    pos += softShadowStrength * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                lightDirection = normalize(pos - hit_point);
            }
            else
            {
                lightDirection = -light.pos;
                if (shadowIntensity > 0.f && softShadowStrength > 0.f)
                    // Soft shadowIntensity
                    lightDirection +=
                        softShadowStrength * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                lightDirection = normalize(lightDirection);
            }
            float nDl = dot(normal, lightDirection);
            const float3 Lc = light.color * light_attenuation;
            float3 directLightingColor = make_float3(0.f);
            const float3 color = make_float3(voxelColor);
            float opacity = voxelColor.w;
            switch (shading_mode)
            {
            case MaterialShadingMode::diffuse:
            case MaterialShadingMode::diffuse_transparency:
            case MaterialShadingMode::perlin:
            {
                float pDl = 1.f;
                if (shading_mode == MaterialShadingMode::perlin)
                {
                    const float3 point = user_parameter * hit_point;
                    const float n1 = 0.25f + 0.75f * clamp(worleyNoise(point, 2.f), 0.f, 1.f);
                    pDl = 1.f - n1;
                    normal.x += 0.5f * n1;
                    normal.y += 0.5f * (0.5f - n1);
                    normal.z += 0.5f * (0.25f - n1);
                    normal = normalize(normal);
                }

                // Diffuse
                directLightingColor += light_attenuation * color * nDl * pDl * Lc;
                const float3 H = normalize(lightDirection - ray.direction);
                const float nDh = dot(normal, H);
                if (nDh > 0.f)
                {
                    // Specular
                    const float power = pow(nDh, phong_exp);
                    directLightingColor += Ks * power * Lc;
                }
                if (shading_mode == MaterialShadingMode::diffuse_transparency)
                    opacity *= nDh;
                break;
            }
            case MaterialShadingMode::cartoon:
            {
                float cosNL = max(0.f, dot(normalize(eye - hit_point), normal));
                const uint angleAsInt = cosNL * user_parameter;
                cosNL = (float)angleAsInt / user_parameter;
                directLightingColor += light_attenuation * color * cosNL * Lc;
                break;
            }
            case MaterialShadingMode::basic:
            {
                const float cosNL = max(0.f, dot(normalize(eye - hit_point), normal));
                directLightingColor += light_attenuation * color * cosNL * Lc;
                break;
            }
            case MaterialShadingMode::electron:
            case MaterialShadingMode::electron_transparency:
            {
                float cosNL = max(0.f, dot(normalize(eye - hit_point), normal));
                cosNL = 1.f - pow(cosNL, user_parameter);
                directLightingColor += light_attenuation * color * cosNL * Lc;
                if (shading_mode == MaterialShadingMode::electron_transparency)
                    opacity *= cosNL;
                break;
            }
            case MaterialShadingMode::checker:
            {
                const int3 point = make_int3(user_parameter * (hit_point + make_float3(1e2f)));
                const int3 p = make_int3(point.x % 2, point.y % 2, point.z % 2);
                if ((p.x == p.y && p.z == 1) || (p.x != p.y && p.z == 0))
                    directLightingColor += light_attenuation * color;
                else
                    directLightingColor += light_attenuation * (1.f - color);
                break;
            }
            case MaterialShadingMode::goodsell:
            {
                const float cosNL = max(0.f, dot(normalize(eye - hit_point), normal));
                directLightingColor += light_attenuation * color * (cosNL > user_parameter ? 1.f : 0.5f);
                break;
            }
            default:
            {
                directLightingColor += light_attenuation * color;
                break;
            }
            }
            voxelColor = make_float4(directLightingColor, opacity);
        }

        // Alpha ratio
        voxelColor = make_float4(make_float3(voxelColor) * DEFAULT_SHADING_ALPHA_RATIO, voxelColor.w);
    }
    const float opacity = voxelColor.w;
    result = make_float3(voxelColor) * light_attenuation * voxelColor.w;

    // Reflection
    const float reflectionIndex = fmaxf(Kr);
    if (reflectionIndex > 0.f && voxelColor.w > 0.f && prd.depth < maxRayDepth)
    {
        PerRayData_radiance reflected_prd;
        reflected_prd.result = make_float4(0.f);
        reflected_prd.importance = prd.importance * reflectionIndex;
        reflected_prd.depth = prd.depth + 1;

        const float3 reflectedNormal = reflect(ray.direction, normal);
        float near = sceneEpsilon + surfaceOffset;
        float far = giRayLength;
        applyClippingPlanes(hit_point, reflectedNormal, near, far);

        const Ray reflected_ray(hit_point, reflectedNormal, radianceRayType, near, far);
        rtTrace(top_object, reflected_ray, reflected_prd);
        result = result * (1.f - Kr) + Kr * make_float3(reflected_prd.result);
    }

    // Refraction
    if (voxelColor.w < 1.f && prd.depth < maxRayDepth)
    {
        PerRayData_radiance refracted_prd;
        refracted_prd.result = make_float4(0.f);
        refracted_prd.importance = prd.importance * (1.f - opacity);
        refracted_prd.depth = prd.depth + 1;

        const float3 refractedNormal = refractedVector(ray.direction, normal, refraction_index, 1.f);
        float near = sceneEpsilon + surfaceOffset;
        float far = giRayLength;
        applyClippingPlanes(hit_point, refractedNormal, near, far);

        const Ray refracted_ray(hit_point, refractedNormal, radianceRayType, near, far);
        rtTrace(top_object, refracted_ray, refracted_prd);
        result = result * opacity + (1.f - opacity) * make_float3(refracted_prd.result);
    }

    // Fog attenuation
    const float z = length(eye - hit_point);
    const float fogAttenuation = z > fogStart ? clamp((z - fogStart) / fogThickness, 0.f, 1.f) : 0.f;
    result = (result * (1.f - fogAttenuation) + fogAttenuation * getEnvironmentColor(ray.direction));

    // Final result
    prd.result = make_float4(result, 1.f);
}

RT_PROGRAM void any_hit_shadow()
{
    prd_shadow.attenuation = 1.f - Ko;
    rtTerminateRay();
}

RT_PROGRAM void closest_hit_radiance()
{
    dicomShade(false);
}

RT_PROGRAM void closest_hit_radiance_textured()
{
    dicomShade(true);
}

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_color(make_float4(0.f, 1.f, 0.f, 1.f));
}
