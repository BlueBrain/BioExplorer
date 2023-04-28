/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

// #include <optix.h>
// #include <optix_world.h>
// #include <optixu/optixu_math_namespace.h>

#include "../../CommonStructs.h"
#include "../Environment.h"
#include "../Helpers.h"
#include "../Random.h"
#include "TransferFunction.h"

#include <brayns/common/CommonTypes.h>

using namespace optix;

const float DEFAULT_VOLUME_SHADOW_THRESHOLD = 0.1f;

struct PerRayData_shadow
{
    float3 attenuation;
};

// System
rtDeclareVariable(float3, bad_color, , );

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

rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );

// Textures
rtDeclareVariable(int, albedoMetallic_map, , );
rtDeclareVariable(float2, texcoord, attribute texcoord, );

// Scene
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(unsigned int, frame, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned int, radianceRayType, , );
rtDeclareVariable(unsigned int, shadowRayType, , );

rtDeclareVariable(float, sceneEpsilon, , );
rtDeclareVariable(float, epsilonFactor, , );

rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float4, jitter4, , );

// Lights
rtBuffer<BasicLight> lights;
rtDeclareVariable(float3, ambientLightColor, , );

// Volume
rtBuffer<unsigned char> volumeData;
rtDeclareVariable(uint3, volumeDimensions, , );
rtDeclareVariable(float3, volumeOffset, , );
rtDeclareVariable(float3, volumeElementSpacing, , );
rtDeclareVariable(uint, volumeSamplesPerRay, , );
rtDeclareVariable(uint, volumeDataTypeSize, , );
// Volume shading
rtDeclareVariable(uint, volumeGradientShadingEnabled, , );
rtDeclareVariable(float, volumeAdaptiveMaxSamplingRate, , );
rtDeclareVariable(uint, volumeAdaptiveSampling, , );
rtDeclareVariable(uint, volumeSingleShade, , );
rtDeclareVariable(uint, volumePreIntegration, , );
rtDeclareVariable(float, volumeSamplingRate, , );
rtDeclareVariable(float3, volumeSpecular, , );

// Simulation data
rtBuffer<float> simulation_data;
rtDeclareVariable(unsigned long, simulation_idx, attribute simulation_idx, );

// Transfer function
rtBuffer<float3> tfColors;
rtBuffer<float> tfOpacities;
rtDeclareVariable(float, tfMinValue, , );
rtDeclareVariable(float, tfRange, , );
rtDeclareVariable(uint, tfSize, , );

// Rendering
rtDeclareVariable(int, maxBounces, , );
rtDeclareVariable(float, shadows, , );
rtDeclareVariable(float, softShadows, , );
rtDeclareVariable(int, softShadowsSamples, , );
rtDeclareVariable(float, mainExposure, , );
rtDeclareVariable(float, giDistance, , );
rtDeclareVariable(float, giWeight, , );
rtDeclareVariable(int, giSamples, , );
rtDeclareVariable(unsigned int, matrixFilter, , );
rtDeclareVariable(float, fogStart, , );
rtDeclareVariable(float, fogThickness, , );

rtBuffer<uchar4, 2> output_buffer;

static __device__ inline bool volumeIntersection(const optix::Ray& ray, float& t0, float& t1)
{
    float3 boxmin = volumeOffset + make_float3(0.f);
    float3 boxmax = volumeOffset + make_float3(volumeDimensions) / volumeElementSpacing;

    float3 a = (boxmin - ray.origin) / ray.direction;
    float3 b = (boxmax - ray.origin) / ray.direction;
    float3 near = fminf(a, b);
    float3 far = fmaxf(a, b);
    t0 = fmaxf(near);
    t1 = fminf(far);

    return (t0 <= t1);
}

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

static __device__ void compose(const float4& src, float4& dst, const float alphaRatio = 1.0)
{
    const float a = alphaRatio * src.w;
    dst = make_float4((1.f - dst.w) * a * make_float3(src) + dst.w * make_float3(dst), dst.w + a);
}

static __device__ float getVoxelValue(const ulong index)
{
    float voxelValue;
    switch (volumeDataTypeSize)
    {
    case 2:
    {
        unsigned char a = volumeData[index * volumeDataTypeSize + 1];
        unsigned char b = volumeData[index * volumeDataTypeSize];
        voxelValue = a * 256 + b;
        break;
    }
    default:
    {
        voxelValue = volumeData[index];
        break;
    }
    }
    return voxelValue;
}

static __device__ float getVolumeShadowContribution(const optix::Ray& volumeRay)
{
    float shadowIntensity = 0.f;
    float t0, t1;
    if (!volumeIntersection(volumeRay, t0, t1))
        return shadowIntensity;

    t0 = max(0.f, t0);
    float tstep = volumeSamplingRate * 4.f;
    float t = t0 + tstep;

    while (t < t1 && shadowIntensity < 1.f)
    {
        const float3 point = volumeRay.origin + volumeRay.direction * t;
        if (point.x > 0.f && point.x < volumeDimensions.x && point.y > 0.f && point.y < volumeDimensions.y &&
            point.z > 0.f && point.z < volumeDimensions.z)
        {
            const ulong index = (long)((ulong)floor(point.x) + (ulong)floor(point.y) * volumeDimensions.x +
                                       (ulong)floor(point.z) * volumeDimensions.x * volumeDimensions.y);
            const float voxelValue = getVoxelValue(index);
            const float4 voxelColor =
                calcTransferFunctionColor(tfMinValue, tfMinValue + tfRange, voxelValue, tfColors, tfOpacities);
            shadowIntensity += voxelColor.w;
        }
        t += tstep;
    }
    return shadowIntensity;
}

static __device__ float4 getVolumeContribution(const optix::Ray& volumeRay)
{
    if (tfColors.size() == 0)
        return make_float4(0.f, 1.f, 0.f, 0.f);

    const uint nbSamples = 7;
    const float3 samples[nbSamples] = {{0, 0, 0}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0, -1}};

    float4 pathColor = make_float4(0.f);

    float t0, t1;
    if (!volumeIntersection(volumeRay, t0, t1))
        return pathColor;

    optix::size_t2 screen = output_buffer.size();
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    float t = max(0.f, t0);
    uint iteration = 0;
    while (t < (t1 - volumeSamplingRate) && pathColor.w < 1.f)
    {
        const float random = rnd(seed) * volumeSamplingRate;
        float4 voxelColor = make_float4(0.f);

        const uint nbSamplesToCompute =
            volumeGradientShadingEnabled ? (volumeSingleShade ? (iteration == 0 ? nbSamples : 1) : nbSamples) : 1;
        uint computedSamples = 0;
        uint computedShadowSamples = 0;
        float shadowIntensity = 0.f;
        for (int i = 0; i < nbSamplesToCompute; ++i)
        {
            const float3 point =
                ((volumeRay.origin + samples[i] * volumeSamplingRate + volumeRay.direction * (t + random)) -
                 volumeOffset) /
                volumeElementSpacing;

            if (point.x > 0.f && point.x < volumeDimensions.x && point.y > 0.f && point.y < volumeDimensions.y &&
                point.z > 0.f && point.z < volumeDimensions.z)
            {
                ++computedSamples;
                const ulong index = (long)((ulong)floor(point.x) + (ulong)floor(point.y) * volumeDimensions.x +
                                           (ulong)floor(point.z) * volumeDimensions.x * volumeDimensions.y);
                const float voxelValue = getVoxelValue(index);
                voxelColor +=
                    calcTransferFunctionColor(tfMinValue, tfMinValue + tfRange, voxelValue, tfColors, tfOpacities);

                // Determine light contribution
                if (computedShadowSamples == 0 && shadows > 0.f && voxelColor.w > DEFAULT_VOLUME_SHADOW_THRESHOLD)
                {
                    ++computedShadowSamples;
                    for (int i = 0; i < lights.size(); ++i)
                    {
                        BasicLight light = lights[i];
                        optix::Ray shadowRay = volumeRay;
                        switch (light.type)
                        {
                        case BASIC_LIGHT_TYPE_POINT:
                        {
                            // Point light
                            float3 lightPosition = light.pos;
                            if (softShadows > 0.f)
                                // Soft shadows
                                lightPosition +=
                                    softShadows * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                            shadowRay.origin = lightPosition;
                            shadowRay.direction = optix::normalize(lightPosition - point);
                            break;
                        }
                        case BASIC_LIGHT_TYPE_DIRECTIONAL:
                        {
                            // Directional light
                            float3 lightDirection = light.dir;
                            if (softShadows > 0.f)
                                // Soft shadows
                                lightDirection +=
                                    softShadows * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                            shadowRay.origin = point;
                            shadowRay.direction = optix::normalize(-1.f * lightDirection);
                            break;
                        }
                        }

                        shadowIntensity += getVolumeShadowContribution(shadowRay);
                    }
                }
            }
        }

        if (computedSamples > 0)
        {
            const float lightAttenuation = 1.f - shadows * shadowIntensity;
            voxelColor.x *= lightAttenuation;
            voxelColor.y *= lightAttenuation;
            voxelColor.z *= lightAttenuation;
            compose(voxelColor / float(computedSamples), pathColor);
        }
        t += volumeSamplingRate;
        ++iteration;
    }

    compose(make_float4(getEnvironmentColor(), 1.f - pathColor.w), pathColor);

    return ::optix::clamp(pathColor, 0.f, 1.f);
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
        if (simulation_data.size() > 0)
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
            if (prd_radiance.depth < maxBounces)
            {
                PerRayData_radiance reflected_prd;
                reflected_prd.depth = prd_radiance.depth + 1;

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
                if (prd_radiance.depth >= maxBounces)
                    continue;

                PerRayData_radiance aa_prd;
                aa_prd.depth = prd_radiance.depth + 1;

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
        if (fmaxf(opacity) == 1.f && prd_radiance.depth < maxBounces)
        {
            // Color bleeding
            if (giWeight > 0.f && prd_radiance.depth == 0)
            {
                PerRayData_radiance new_prd;
                new_prd.depth = prd_radiance.depth + 1;

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
    if (fmaxf(opacity) < 1.f && prd_radiance.depth < maxBounces)
    {
        PerRayData_radiance refracted_prd;
        refracted_prd.depth = prd_radiance.depth + 1;

        const float3 R = refractedVector(ray.direction, normal, p_refractionIndex, 1.f);
        const optix::Ray refracted_ray(hit_point, R, radianceRayType, epsilon, giDistance);
        rtTrace(top_object, refracted_ray, refracted_prd);
        color = color * opacity + (1.f - opacity) * refracted_prd.result;
    }

    float4 finalColor = make_float4(color, fmaxf(opacity));

    // Volume
    if (volumeData.size() > 0)
    {
        const float4 volumeColor = getVolumeContribution(ray);
        compose(volumeColor, finalColor);
    }
    float3 result = make_float3(finalColor);

    // Matrix filter :)
    if (matrixFilter)
        result = make_float3(result.x * 0.666f, result.y * 0.8f, result.z * 0.666f);

    // Exposure and Fog attenuation
    const float z = optix::length(eye - hit_point);
    const float fogAttenuation = z > fogStart ? optix::clamp((z - fogStart) / fogThickness, 0.f, 1.f) : 0.f;
    result = mainExposure * (result * (1.f - fogAttenuation) + fogAttenuation * getEnvironmentColor());

    prd_radiance.result = result;
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
