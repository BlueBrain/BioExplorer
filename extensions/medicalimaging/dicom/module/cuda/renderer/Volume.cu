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

#include <platform/engines/optix6/OptiXCommonStructs.h>
#include <platform/engines/optix6/cuda/Environment.cuh>
#include <platform/engines/optix6/cuda/Helpers.cuh>
#include <platform/engines/optix6/cuda/Random.cuh>

#include <platform/engines/optix6/cuda/renderer/TransferFunction.cuh>

#include <platform/core/common/CommonTypes.h>

using namespace optix;

const float DEFAULT_VOLUME_SHADOW_THRESHOLD = 0.1f;

// System
rtDeclareVariable(float3, bad_color, , );

// Material attributes
rtDeclareVariable(float3, Ka, , );
rtDeclareVariable(float3, Kd, , );
rtDeclareVariable(float3, Ks, , );
rtDeclareVariable(float3, Kr, , );
rtDeclareVariable(float3, Ko, , );
rtDeclareVariable(float, phong_exp, , );
rtDeclareVariable(uint, shading_mode, , );

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

rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float4, jitter4, , );

// Lights
rtBuffer<BasicLight> lights;
rtDeclareVariable(float3, ambientLightColor, , );

// Volume
rtDeclareVariable(uint3, volumeDimensions, , );
rtDeclareVariable(float3, volumeOffset, , );
rtDeclareVariable(float3, volumeElementSpacing, , );
rtDeclareVariable(uint, volumeSamplesPerRay, , );
rtDeclareVariable(uint, volumeDataTypeSize, , );
rtDeclareVariable(uint, volumeDataType, , );
rtDeclareVariable(int, volumeSampler, , );

// Volume shading
rtDeclareVariable(uint, volumeGradientShadingEnabled, , );
rtDeclareVariable(float, volumeAdaptiveMaxSamplingRate, , );
rtDeclareVariable(uint, volumeAdaptiveSampling, , );
rtDeclareVariable(uint, volumeSingleShade, , );
rtDeclareVariable(uint, volumePreIntegration, , );
rtDeclareVariable(float, volumeSamplingRate, , );
rtDeclareVariable(float3, volumeSpecular, , );

// Transfer function
rtBuffer<float3> tfColors;
rtBuffer<float> tfOpacities;
rtDeclareVariable(float, tfMinValue, , );
rtDeclareVariable(float, tfRange, , );
rtDeclareVariable(uint, tfSize, , );

// Rendering
rtDeclareVariable(float, shadows, , );
rtDeclareVariable(float, softShadows, , );
rtDeclareVariable(float, mainExposure, , );
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

static __device__ void compose(const float4& src, float4& dst, const float alphaRatio = 1.0)
{
    const float a = alphaRatio * src.w;
    dst = make_float4((1.f - dst.w) * a * make_float3(src) + dst.w * make_float3(dst), dst.w + a);
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
            const unsigned int voxelValue =
                optix::rtTex3D<unsigned int>(volumeSampler, point.x / volumeDimensions.x / 2.f,
                                             point.y / volumeDimensions.y / 2.f, point.z / volumeDimensions.z / 2.f);
            const float4 voxelColor = calcTransferFunctionColor(tfMinValue, tfMinValue + tfRange,
                                                                float(voxelValue) / 65536.f, tfColors, tfOpacities);

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

            if (point.x >= 0.f && point.x < volumeDimensions.x && point.y >= 0.f && point.y < volumeDimensions.y &&
                point.z >= 0.f && point.z < volumeDimensions.z)
            {
                ++computedSamples;
                const unsigned int voxelValue =
                    optix::rtTex3D<unsigned int>(volumeSampler, point.x / volumeDimensions.x / 2.f,
                                                 point.y / volumeDimensions.y / 2.f,
                                                 point.z / volumeDimensions.z / 2.f);
                voxelColor += calcTransferFunctionColor(tfMinValue, tfMinValue + tfRange, float(voxelValue) / 65536.f,
                                                        tfColors, tfOpacities);

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

static __device__ void volumeShadowed(float3 p_Ko)
{
    // this material is opaque, so it fully attenuates all shadow rays
    prd_shadow.attenuation = 1.f - p_Ko;
    rtTerminateRay();
}

static __device__ void volumeShade()
{
    const float3 hit_point = ray.origin + t_hit * ray.direction;
    float4 color = make_float4(0.f);

    // Volume
    const float4 volumeColor = getVolumeContribution(ray);
    compose(volumeColor, color);
    float3 result = make_float3(color);

    // Exposure and Fog attenuation
    const float z = optix::length(eye - hit_point);
    const float fogAttenuation = z > fogStart ? optix::clamp((z - fogStart) / fogThickness, 0.f, 1.f) : 0.f;
    result = mainExposure * (result * (1.f - fogAttenuation) + fogAttenuation * getEnvironmentColor());

    prd_radiance.result = result;
}

RT_PROGRAM void any_hit_shadow()
{
    volumeShadowed(Ko);
}

static __device__ inline void shade(bool textured)
{
    volumeShade();
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
