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

rtDeclareVariable(float, epsilonFactor, , );
rtDeclareVariable(int, maxBounces, , );
rtDeclareVariable(int, softShadowsSamples, , );
rtDeclareVariable(unsigned int, matrixFilter, , );

static __device__ inline float3 frac(const float3 x)
{
    return x - optix::floor(x);
}

static __device__ inline float mix(const float x, const float y, const float a)
{
    return x * (1.f - a) + y * a;
}

static __device__ inline float hash(float n)
{
    return frac(make_float3(sin(n + 1.951f) * 43758.5453f)).x;
}

static __device__ float noise(const float3& x)
{
    // hash based 3d value noise
    float3 p = optix::floor(x);
    float3 f = frac(x);

    f = f * f * (make_float3(3.0f) - make_float3(2.0f) * f);
    float n = p.x + p.y * 57.0f + 113.0f * p.z;
    return mix(mix(mix(hash(n + 0.0f), hash(n + 1.0f), f.x), mix(hash(n + 57.0f), hash(n + 58.0f), f.x), f.y),
               mix(mix(hash(n + 113.0f), hash(n + 114.0f), f.x), mix(hash(n + 170.0f), hash(n + 171.0f), f.x), f.y),
               f.z);
}

static __device__ inline float3 mod(const float3& v, const int m)
{
    return make_float3(v.x - m * floor(v.x / m), v.y - m * floor(v.y / m), v.z - m * floor(v.z / m));
}

static __device__ float cells(const float3& p, float cellCount)
{
    const float3 pCell = p * cellCount;
    float d = 1.0e10;
    for (int xo = -1; xo <= 1; xo++)
    {
        for (int yo = -1; yo <= 1; yo++)
        {
            for (int zo = -1; zo <= 1; zo++)
            {
                float3 tp = floor(pCell) + make_float3(xo, yo, zo);

                tp = pCell - tp - noise(mod(tp, cellCount / 1));

                d = min(d, optix::dot(tp, tp));
            }
        }
    }
    d = min(d, 1.0f);
    d = max(d, 0.0f);
    return d;
}

static __device__ float worleyNoise(const float3& p, float cellCount)
{
    return cells(p, cellCount);
}

static __device__ float3 refractedVector(const float3 direction, const float3 normal, const float n1, const float n2)
{
    if (n2 == 0.f)
        return direction;
    const float eta = n1 / n2;
    const float cos1 = -optix::dot(direction, normal);
    const float cos2 = 1.f - eta * eta * (1.f - cos1 * cos1);
    if (cos2 > 0.f)
        return ::optix::normalize(eta * direction + (eta * cos1 - sqrt(cos2)) * normal);
    return direction;
}

static __device__ void phongShadowed(float3 p_Ko)
{
    // this material is opaque, so it fully attenuates all shadow rays
    prd_shadow.attenuation = 1.f - p_Ko;
    rtTerminateRay();
}

static __device__ void phongShade(float3 p_Kd, float3 p_Ka, float3 p_Ks, float3 p_Kr, float3 p_Ko,
                                  float p_refractionIndex, float p_phong_exp, float p_glossiness,
                                  unsigned int p_shadingMode, float p_user_parameter, float3 p_normal)
{
    const float3 hit_point = ray.origin + t_hit * ray.direction;
    float3 color = make_float3(0.f, 0.f, 0.f);
    float3 opacity = p_Ko;
    float3 Kd = p_Kd;

    float3 normal = ::optix::normalize(p_normal);
    const float epsilon = sceneEpsilon * epsilonFactor * optix::length(eye - hit_point);
    if (fmaxf(opacity) > 0.f)
    {
        // User data
        if (cast_user_data && simulation_data.size() > 0)
        {
            const float4 userDataColor =
                calcTransferFunctionColor(tfMinValue, tfMinValue + tfRange, simulation_data[simulation_idx], tfColors,
                                          tfOpacities);
            Kd = Kd * (1.f - userDataColor.w) + make_float3(userDataColor) * userDataColor.w;
        }

        const float userParameter = p_user_parameter;

        // Randomness
        optix::size_t2 screen = output_buffer.size();
        unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

        // Glossiness
        if (p_glossiness < 1.f)
            normal = optix::normalize(normal + (1.f - p_glossiness) *
                                                   make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));

        // compute direct lighting
        float3 directLightingColor = make_float3(0.f, 0.f, 0.f);
        unsigned int num_lights = lights.size();
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
                if (shadows > 0.f && softShadows > 0.f)
                    // Soft shadows
                    pos += softShadows * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                lightDirection = optix::normalize(pos - hit_point);
            }
            else
            {
                // Directional light
                lightDirection = -light.pos;
                if (shadows > 0.f && softShadows > 0.f)
                    // Soft shadows
                    lightDirection += softShadows * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                lightDirection = optix::normalize(lightDirection);
            }
            float nDl = optix::dot(normal, lightDirection);

            // Shadows
            if (shadows > 0.f)
            {
                if (nDl > 0.f && light.casts_shadow)
                {
                    PerRayData_shadow shadow_prd;
                    shadow_prd.attenuation = make_float3(1.f);
                    optix::Ray shadow_ray(hit_point, lightDirection, shadowRayType, epsilon, giDistance);
                    rtTrace(top_shadower, shadow_ray, shadow_prd);

                    // light_attenuation is zero if completely shadowed
                    light_attenuation -= shadows * (1.f - ::optix::luminance(shadow_prd.attenuation));
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
                    directLightingColor += light_attenuation * Kd * nDl * pDl * Lc;
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
                    directLightingColor += light_attenuation * Kd * cosNL * Lc;
                    break;
                }
                case MaterialShadingMode::basic:
                {
                    const float cosNL = optix::max(0.f, optix::dot(optix::normalize(eye - hit_point), normal));
                    directLightingColor += light_attenuation * Kd * cosNL * Lc;
                    break;
                }
                case MaterialShadingMode::electron:
                case MaterialShadingMode::electron_transparency:
                {
                    float cosNL = max(0.f, optix::dot(optix::normalize(eye - hit_point), normal));
                    cosNL = 1.f - pow(cosNL, userParameter);
                    directLightingColor += light_attenuation * Kd * cosNL * Lc;
                    if (p_shadingMode == MaterialShadingMode::electron_transparency)
                        opacity *= cosNL;
                    break;
                }
                case MaterialShadingMode::checker:
                {
                    const int3 point = make_int3(userParameter * (hit_point + make_float3(1e2f)));
                    const int3 p = make_int3(point.x % 2, point.y % 2, point.z % 2);
                    if ((p.x == p.y && p.z == 1) || (p.x != p.y && p.z == 0))
                        directLightingColor += light_attenuation * Kd;
                    else
                        directLightingColor += light_attenuation * (1.f - Kd);
                    break;
                }
                case MaterialShadingMode::goodsell:
                {
                    const float cosNL = max(0.f, optix::dot(optix::normalize(eye - hit_point), normal));
                    directLightingColor += light_attenuation * Kd * (cosNL > userParameter ? 1.f : 0.5f);
                    break;
                }
                default:
                {
                    directLightingColor += light_attenuation * Kd;
                    break;
                }
                }
            }
        }
        color += directLightingColor;

        // Reflection
        if (fmaxf(p_Kr) > 0.f)
        {
            if (prd.depth < maxBounces)
            {
                PerRayData_radiance reflected_prd;
                reflected_prd.depth = prd.depth + 1;

                const float3 R = optix::reflect(ray.direction, normal);
                const optix::Ray reflected_ray(hit_point, R, radianceRayType, epsilon, giDistance);
                rtTrace(top_object, reflected_ray, reflected_prd);
                color = color * (1.f - p_Kr) + p_Kr * reflected_prd.result;
            }
        }

        // Ambient occlusion
        if (giSamples > 0 && giWeight > 0.f)
        {
            float3 aa_color = make_float3(0.f);
            for (int i = 0; i < giSamples; ++i)
            {
                if (prd.depth >= maxBounces)
                    continue;

                PerRayData_radiance aa_prd;
                aa_prd.depth = prd.depth + 1;

                float3 aa_normal = optix::normalize(make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));
                if (optix::dot(aa_normal, normal) < 0.f)
                    aa_normal = -aa_normal;

                const optix::Ray aa_ray(hit_point, aa_normal, shadowRayType, epsilon, giDistance);
                rtTrace(top_object, aa_ray, aa_prd);
                aa_color = aa_color + giWeight * aa_prd.result;
            }
            color += aa_color / giSamples;
        }

        // Only opaque surfaces are affected by Global Illumination
        if (fmaxf(opacity) == 1.f && prd.depth < maxBounces)
        {
            // Color bleeding
            if (giWeight > 0.f && prd.depth == 0)
            {
                PerRayData_radiance new_prd;
                new_prd.depth = prd.depth + 1;

                float3 ra_normal =
                    ::optix::normalize(make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));
                if (optix::dot(ra_normal, normal) < 0.f)
                    ra_normal = -ra_normal;

                const float3 origin = hit_point + epsilonFactor * ra_normal;
                const optix::Ray ra_ray = optix::make_Ray(origin, ra_normal, radianceRayType, epsilon, ray.tmax);
                rtTrace(top_shadower, ra_ray, new_prd);
                color += giWeight * new_prd.result;
            }
        }
    }

    // Refraction
    if (fmaxf(opacity) < 1.f && prd.depth < maxBounces)
    {
        PerRayData_radiance refracted_prd;
        refracted_prd.depth = prd.depth + 1;

        const float3 R = refractedVector(ray.direction, normal, p_refractionIndex, 1.f);
        const optix::Ray refracted_ray(hit_point, R, radianceRayType, epsilon, giDistance);
        rtTrace(top_object, refracted_ray, refracted_prd);
        color = color * opacity + (1.f - opacity) * refracted_prd.result;
    }

    float4 finalColor = make_float4(color, fmaxf(opacity));

    // Volume
    const float4 volumeColor = getVolumeContribution(ray);
    compose(volumeColor, finalColor);
    float3 result = make_float3(finalColor);

    // Matrix filter :)
    if (matrixFilter)
        result = make_float3(result.x * 0.666f, result.y * 0.8f, result.z * 0.666f);

    // Exposure and Fog attenuation
    const float z = optix::length(eye - hit_point);
    const float fogAttenuation = z > fogStart ? optix::clamp((z - fogStart) / fogThickness, 0.f, 1.f) : 0.f;
    result = mainExposure * (result * (1.f - fogAttenuation) + fogAttenuation * getEnvironmentColor());

    prd.result = result;
}

RT_PROGRAM void any_hit_shadow()
{
    phongShadowed(Ko);
}

static __device__ inline void shade(bool textured)
{
    float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
    float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));

    float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

    float3 p_Kd = Kd;
    if (textured)
        p_Kd = make_float3(optix::rtTex2D<float4>(albedoMetallic_map, texcoord.x, texcoord.y));

    phongShade(p_Kd, Ka, Ks, Kr, Ko, refraction_index, phong_exp, glossiness, shading_mode, user_parameter, ffnormal);
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
