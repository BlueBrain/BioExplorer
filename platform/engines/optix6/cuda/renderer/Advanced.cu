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

rtDeclareVariable(float, epsilonMultiplier, , );
rtDeclareVariable(int, shadowSamples, , );
rtDeclareVariable(uint, matrixFilter, , );

static __device__ void phongShadowed(float3 p_Ko)
{
    // this material is opaque, so it fully attenuates all shadow rays
    prd_shadow.attenuation = 1.f - p_Ko;
    rtTerminateRay();
}

static __device__ void phongShade(float3 p_Kd, float3 p_Ka, float3 p_Ks, float3 p_Kr, float3 p_Ko,
                                  float p_refractionIndex, float p_phong_exp, float p_glossiness, uint p_shadingMode,
                                  float p_user_parameter, float3 p_normal)
{
    const float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 color = make_float3(0.f);
    float3 opacity = p_Ko;
    float3 l_Kd = p_Kd;

    float3 normal = ::optix::normalize(p_normal);
    if (fmaxf(opacity) > 0.f && prd.depth < maxRayDepth)
    {
        // User data
        const float4 userDataColor = getUserData();
        l_Kd = l_Kd * (1.f - userDataColor.w) + make_float3(userDataColor) * userDataColor.w;

        const float userParameter = p_user_parameter;

        // Randomness
        optix::size_t2 screen = output_buffer.size();
        uint seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

        // Glossiness
        if (p_glossiness < 1.f)
            normal = optix::normalize(normal + (1.f - p_glossiness) *
                                                   make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));

        // compute direct lighting
        float3 directLightingColor = make_float3(0.f, 0.f, 0.f);
        uint num_lights = lights.size();
        for (int i = 0; i < num_lights; ++i)
        {
            // Surface
            float light_attenuation = 1.f;

            BasicLight light = lights[i];
            float3 lightDirection;

            if (light.type == BASIC_LIGHT_TYPE_POINT)
            {
                // Point light
                float3 pos = light.pos;
                if (shadowIntensity > 0.f && softShadowStrength > 0.f)
                    // Soft shadowIntensity
                    pos += softShadowStrength * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                lightDirection = optix::normalize(pos - hit_point);
            }
            else
            {
                // Directional light
                lightDirection = -light.pos;
                if (shadowIntensity > 0.f && softShadowStrength > 0.f)
                    // Soft shadowIntensity
                    lightDirection +=
                        softShadowStrength * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                lightDirection = optix::normalize(lightDirection);
            }
            float nDl = optix::dot(normal, lightDirection);

            // Shadows
            if (shadowIntensity > 0.f)
            {
                if (nDl > 0.f && light.casts_shadow)
                {
                    PerRayData_shadow shadow_prd;
                    shadow_prd.attenuation = make_float3(1.f);
                    optix::Ray shadow_ray(hit_point, lightDirection, shadowRayType, sceneEpsilon, giRayLength);
                    rtTrace(top_shadower, shadow_ray, shadow_prd);

                    // light_attenuation is zero if completely shadowed
                    light_attenuation -= shadowIntensity * (1.f - ::optix::luminance(shadow_prd.attenuation));
                }
            }

            // If not completely shadowed, light the hit point
            if (light_attenuation > 0.f)
            {
                const float3 Lc = light.color * light_attenuation;
                switch (p_shadingMode)
                {
                case MaterialShadingMode::diffuse:
                case MaterialShadingMode::diffuse_transparency:
                case MaterialShadingMode::perlin:
                {
                    float pDl = 1.f;
                    if (p_shadingMode == MaterialShadingMode::perlin)
                    {
                        const float3 point = userParameter * hit_point;
                        const float n1 = 0.25f + 0.75f * optix::clamp(worleyNoise(point, 2.f), 0.f, 1.f);
                        pDl = 1.f - n1;
                        normal.x += 0.5f * n1;
                        normal.y += 0.5f * (0.5f - n1);
                        normal.z += 0.5f * (0.25f - n1);
                        normal = optix::normalize(normal);
                    }

                    // Diffuse
                    directLightingColor += light_attenuation * l_Kd * nDl * pDl * Lc;
                    const float3 H = optix::normalize(lightDirection - ray.direction);
                    const float nDh = optix::dot(normal, H);
                    if (nDh > 0.f)
                    {
                        // Specular
                        const float power = pow(nDh, p_phong_exp);
                        directLightingColor += p_Ks * power * Lc;
                    }
                    if (p_shadingMode == MaterialShadingMode::diffuse_transparency)
                        opacity *= nDh;
                    break;
                }
                case MaterialShadingMode::cartoon:
                {
                    float cosNL = max(0.f, optix::dot(optix::normalize(eye - hit_point), normal));
                    const uint angleAsInt = cosNL * userParameter;
                    cosNL = (float)angleAsInt / userParameter;
                    directLightingColor += light_attenuation * l_Kd * cosNL * Lc;
                    break;
                }
                case MaterialShadingMode::basic:
                {
                    const float cosNL = optix::max(0.f, optix::dot(optix::normalize(eye - hit_point), normal));
                    directLightingColor += light_attenuation * l_Kd * cosNL * Lc;
                    break;
                }
                case MaterialShadingMode::electron:
                case MaterialShadingMode::electron_transparency:
                {
                    float cosNL = max(0.f, optix::dot(optix::normalize(eye - hit_point), normal));
                    cosNL = 1.f - pow(cosNL, userParameter);
                    directLightingColor += light_attenuation * l_Kd * cosNL * Lc;
                    if (p_shadingMode == MaterialShadingMode::electron_transparency)
                        opacity *= cosNL;
                    break;
                }
                case MaterialShadingMode::checker:
                {
                    const int3 point = make_int3(userParameter * (hit_point + make_float3(1e2f)));
                    const int3 p = make_int3(point.x % 2, point.y % 2, point.z % 2);
                    if ((p.x == p.y && p.z == 1) || (p.x != p.y && p.z == 0))
                        directLightingColor += light_attenuation * l_Kd;
                    else
                        directLightingColor += light_attenuation * (1.f - l_Kd);
                    break;
                }
                case MaterialShadingMode::goodsell:
                {
                    const float cosNL = max(0.f, optix::dot(optix::normalize(eye - hit_point), normal));
                    directLightingColor += light_attenuation * l_Kd * (cosNL > userParameter ? 1.f : 0.5f);
                    break;
                }
                case MaterialShadingMode::surface_normal:
                {
                    directLightingColor += light_attenuation * (0.5f + 0.5f * normal);
                    break;
                }
                default:
                {
                    directLightingColor += light_attenuation * l_Kd;
                    break;
                }
                }
            }
        }
        color += directLightingColor;

        // Reflection
        const float reflection = fmaxf(p_Kr);
        if (reflection > 0.f && prd.depth < maxRayDepth - 1)
        {
            PerRayData_radiance reflected_prd;
            reflected_prd.result = make_float4(0.f);
            reflected_prd.importance = prd.importance * fmaxf(p_Kr);
            reflected_prd.depth = prd.depth + 1;

            const float3 R = optix::reflect(ray.direction, normal);
            const optix::Ray reflected_ray(hit_point, R, radianceRayType, sceneEpsilon, giRayLength);
            rtTrace(top_object, reflected_ray, reflected_prd);
            color = color * (1.f - reflection) + Kr * make_float3(reflected_prd.result);
        }

        // Refraction
        if (fmaxf(opacity) < 1.f && prd.depth < maxRayDepth - 1)
        {
            PerRayData_radiance refracted_prd;
            refracted_prd.result = make_float4(0.f);
            refracted_prd.importance = prd.importance * (1.f - fmaxf(opacity));
            refracted_prd.depth = prd.depth + 1;

            const float3 refractedNormal = refractedVector(ray.direction, normal, p_refractionIndex, 1.f);
            const optix::Ray refracted_ray(hit_point, refractedNormal, radianceRayType, sceneEpsilon, giRayLength);
            rtTrace(top_object, refracted_ray, refracted_prd);
            color = color * opacity + (1.f - opacity) * make_float3(refracted_prd.result);
        }

        // Ambient occlusion
        if (giSamples > 0 && giWeight > 0.f && prd.depth == 0)
        {
            float aa_attenuation = 0.f;
            float3 cb_color = make_float3(0.f);
            for (uint i = 0; i < giSamples; ++i)
            {
                // Ambient occlusion
                PerRayData_shadow aa_prd;
                aa_prd.attenuation = make_float3(0.f);

                float3 aa_normal = optix::normalize(make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));
                if (optix::dot(aa_normal, normal) < 0.f)
                    aa_normal = -aa_normal;

                const optix::Ray aa_ray(hit_point, aa_normal, shadowRayType, sceneEpsilon, giRayLength);
                rtTrace(top_object, aa_ray, aa_prd);
                aa_attenuation += giWeight * ::optix::luminance(aa_prd.attenuation);

                // Color bleeding
                PerRayData_radiance cb_prd;
                cb_prd.result = make_float4(0.f);
                cb_prd.importance = 0.f;
                cb_prd.depth = prd.depth + 1;

                float3 cb_normal =
                    ::optix::normalize(make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));
                if (optix::dot(cb_normal, normal) < 0.f)
                    cb_normal = -cb_normal;

                const optix::Ray cb_ray =
                    optix::make_Ray(hit_point, cb_normal, radianceRayType, sceneEpsilon, ray.tmax);
                rtTrace(top_shadower, cb_ray, cb_prd);
                cb_color += giWeight * make_float3(cb_prd.result);
            }
            aa_attenuation /= (float)giSamples;
            cb_color /= (float)giSamples;
            color += cb_color * (1.f - aa_attenuation);
        }
    }

    float4 finalColor = make_float4(color, ::optix::luminance(p_Ko));

    float3 result = make_float3(finalColor);

    // Matrix filter :)
    if (matrixFilter)
        result = make_float3(result.x * 0.666f, result.y * 0.8f, result.z * 0.666f);

    // Fog attenuation
    const float z = optix::length(eye - hit_point);
    const float fogAttenuation = z > fogStart ? optix::clamp((z - fogStart) / fogThickness, 0.f, 1.f) : 0.f;
    result = result * (1.f - fogAttenuation) + fogAttenuation * getEnvironmentColor(ray.direction);

    prd.result = make_float4(result, finalColor.w);
}

RT_PROGRAM void any_hit_shadow()
{
    phongShadowed(Ko);
}

static __device__ inline void shade(bool textured)
{
    const float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    const float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
    const float3 normal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

    float3 p_Kd = Kd;
    float3 p_Ko = Ko;
    if (textured)
    {
        if (volume_map != 0)
        {
            const float voxelValue = optix::rtTex3D<float>(volume_map, texcoord3d.x, texcoord3d.y, texcoord3d.z);
            const float4 voxelColor = calcTransferFunctionColor(transfer_function_map, value_range, voxelValue);
            p_Kd = make_float3(voxelColor);
            p_Ko = make_float3(voxelColor.w);
        }
        else if (octree_indices_map != 0 && octree_values_map != 0)
        {
            const float4 voxelColor = calcTransferFunctionColor(transfer_function_map, value_range, texcoord.x);
            p_Kd = make_float3(voxelColor);
            p_Ko = make_float3(voxelColor.w);
        }
        else
            p_Kd = make_float3(optix::rtTex2D<float4>(albedoMetallic_map, texcoord.x, texcoord.y));
    }

    phongShade(p_Kd, Ka, Ks, Kr, p_Ko, refraction_index, phong_exp, glossiness, shading_mode, user_parameter, normal);
}

RT_PROGRAM void closest_hit_radiance()
{
    shade(false);
}

RT_PROGRAM void closest_hit_radiance_textured()
{
    shade(true);
}

RT_PROGRAM void exception()
{
    output_buffer[launch_index] = make_color(bad_color);
}
