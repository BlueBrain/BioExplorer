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

#include <platform/engines/optix6/cuda/Helpers.cuh>
#include <platform/engines/optix6/cuda/renderer/TransferFunction.cuh>

#include <platform/core/common/CommonTypes.h>

using namespace optix;

const float DEFAULT_VOLUME_SHADOW_THRESHOLD = 0.1f;

// Rendering
rtDeclareVariable(float, shadows, , );
rtDeclareVariable(float, softShadows, , );
rtDeclareVariable(float, fogStart, , );
rtDeclareVariable(float, fogThickness, , );
rtDeclareVariable(int, giSamples, , );
rtDeclareVariable(float, giWeight, , );
rtDeclareVariable(float, giDistance, , );
rtDeclareVariable(unsigned int, shadingEnabled, , );
rtDeclareVariable(float, specularExponent, , );

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

static __device__ void compose(const float4& src, float4& dst, const float alphaCorrection = 1.0)
{
    const float alpha = alphaCorrection * src.w;
    dst =
        make_float4((1.f - dst.w) * alpha * make_float3(src) + dst.w * make_float3(dst), dst.w + alpha * (1.f - dst.w));
}

static __device__ float getVoxelValue(const float3& p)
{
    switch (volumeDataType)
    {
    case RT_FORMAT_BYTE:
    {
        const char voxelValue = optix::rtTex3D<char>(volumeSampler, p.x, p.y, p.z);
        return float(voxelValue) / 256.f;
    }
    case RT_FORMAT_UNSIGNED_BYTE:
    {
        const unsigned char voxelValue = optix::rtTex3D<unsigned char>(volumeSampler, p.x, p.y, p.z);
        return float(voxelValue) / 256.f;
    }
    case RT_FORMAT_INT:
    {
        const int voxelValue = optix::rtTex3D<int>(volumeSampler, p.x, p.y, p.z);
        return float(voxelValue) / 65536.f;
    }
    case RT_FORMAT_UNSIGNED_INT:
    {
        const unsigned int voxelValue = optix::rtTex3D<unsigned int>(volumeSampler, p.x, p.y, p.z);
        return float(voxelValue) / 65536.f;
    }
    default:
    {
        return optix::rtTex3D<float>(volumeSampler, p.x, p.y, p.z);
    }
    }
}

static __device__ float getVolumeShadowContribution(const optix::Ray& volumeRay, const float limit = 1.0)
{
    float shadowIntensity = 0.f;
    float t0, t1;
    if (!volumeIntersection(volumeRay, t0, t1))
        return shadowIntensity;

    applyClippingPlanes(volumeRay.origin, volumeRay.direction, t0, t1);

    optix::size_t2 screen = output_buffer.size();
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    t0 = max(0.f, t0);
    const float tstep = volumeSamplingRate;
    const float random = rnd(seed) * tstep;
    float t = t0 + tstep + random;
    float distance = 0.f;

    while (t < t1 && shadowIntensity < limit && distance < giDistance)
    {
        const float3 point =
            ((volumeRay.origin + volumeSamplingRate + volumeRay.direction * t) - volumeOffset) / volumeElementSpacing;

        if (point.x > 0.f && point.x < volumeDimensions.x && point.y > 0.f && point.y < volumeDimensions.y &&
            point.z > 0.f && point.z < volumeDimensions.z)
        {
            const float3 p = make_float3(point.x / volumeDimensions.x / 2.f, point.y / volumeDimensions.y / 2.f,
                                         point.z / volumeDimensions.z / 2.f);
            const float4 voxelColor =
                calcTransferFunctionColor(tfMinValue, tfMinValue + tfRange, getVoxelValue(p), tfColors, tfOpacities);

            shadowIntensity += voxelColor.w;
        }
        t += tstep;
        distance += tstep;
    }
    return shadowIntensity;
}

static __device__ float4 getVolumeContribution(const optix::Ray& volumeRay)
{
    if (tfColors.size() == 0)
        return make_float4(0.f, 1.f, 0.f, 0.f);

    float4 pathColor = make_float4(0.f, 0.f, 0.f, 0.f);

    float t0, t1;
    if (!volumeIntersection(volumeRay, t0, t1))
        return pathColor;

    applyClippingPlanes(volumeRay.origin, volumeRay.direction, t0, t1);

    optix::size_t2 screen = output_buffer.size();
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);
    const float random = rnd(seed) * volumeSamplingRate;

    float t = max(0.f, t0) + random;
    bool giShading = true;
    while (t < (t1 - volumeSamplingRate) && pathColor.w < 1.f)
    {
        const float3 point =
            ((volumeRay.origin + volumeSamplingRate + volumeRay.direction * (t + random)) - volumeOffset) /
            volumeElementSpacing;

        float4 voxelColor = make_float4(0.f);
        float shadowIntensity = 0.f;
        float aaIntensity = 0.f;

        if (point.x >= 0.f && point.x < volumeDimensions.x && point.y >= 0.f && point.y < volumeDimensions.y &&
            point.z >= 0.f && point.z < volumeDimensions.z)
        {
            const float3 p = make_float3(point.x / volumeDimensions.x / 2.f, point.y / volumeDimensions.y / 2.f,
                                         point.z / volumeDimensions.z / 2.f);

            const float voxelValue = getVoxelValue(p);
            voxelColor +=
                calcTransferFunctionColor(tfMinValue, tfMinValue + tfRange, voxelValue, tfColors, tfOpacities);

            // Determine light contribution
            if (shadows > 0.f && voxelColor.w > DEFAULT_VOLUME_SHADOW_THRESHOLD && giShading)
            {
                const float3 hit_point = ray.origin + t * ray.direction;
                giShading = (volumeSingleShade ? false : true);
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
                        shadowRay.direction = optix::normalize(lightPosition - hit_point);
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
                        shadowRay.origin = hit_point;
                        shadowRay.direction = -optix::normalize(lightDirection);
                        break;
                    }
                    }

                    shadowIntensity += getVolumeShadowContribution(shadowRay) * shadows;
                }
            }

            // Ambient occlusion
            for (int i = 0; i < giSamples && voxelColor.w > DEFAULT_VOLUME_SHADOW_THRESHOLD && giShading; ++i)
            {
                const float3 aa_normal =
                    optix::normalize(make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));
                optix::Ray aa_ray = volumeRay;
                aa_ray.origin = point;
                aa_ray.direction = aa_normal;
                aaIntensity += getVolumeShadowContribution(aa_ray) * giWeight;
            }
            if (giSamples > 0)
                aaIntensity /= float(giSamples);

            // Shading
            if (shadingEnabled && voxelValue > DEFAULT_VOLUME_SHADOW_THRESHOLD)
            {
                float3 normal = make_float3(0.f);
                const float3 positions[6] = {
                    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
                };
                for (uint i = 0; i < 6; ++i)
                {
                    const float3 p1 = point + positions[i] * volumeNormalEpsilon;
                    const float3 p2 = make_float3(p1.x / volumeDimensions.x / 2.f, p1.y / volumeDimensions.y / 2.f,
                                                  p1.z / volumeDimensions.z / 2.f);

                    const float value = getVoxelValue(p2);
                    if (value > DEFAULT_VOLUME_SHADOW_THRESHOLD)
                        normal = normal + value * positions[i];
                }
                normal = ::optix::normalize(normal);

                for (int i = 0; i < lights.size(); ++i)
                {
                    BasicLight light = lights[i];
                    float3 L;
                    switch (light.type)
                    {
                    case BASIC_LIGHT_TYPE_POINT:
                    {
                        // Point light
                        L = optix::normalize(light.pos - point);
                        break;
                    }
                    case BASIC_LIGHT_TYPE_DIRECTIONAL:
                    {
                        L = optix::normalize(light.dir);
                        break;
                    }
                    }
                    const float cosNL = clamp(dot(normal, L), 0.f, 1.f);
                    if (cosNL > 0.f)
                    {
                        const float power = pow(cosNL, specularExponent);
                        voxelColor =
                            make_float4(make_float3(voxelColor) * cosNL + power * make_float3(1, 1, 1), voxelColor.w);
                    }
                    else
                        voxelColor = make_float4(make_float3(voxelColor) * cosNL, voxelColor.w);
                }
            }

            const float lightAttenuation = clamp(1.f - (shadowIntensity + aaIntensity) * voxelColor.w, 0.f, 1.f);
            voxelColor.x *= lightAttenuation;
            voxelColor.y *= lightAttenuation;
            voxelColor.z *= lightAttenuation;
            compose(voxelColor, pathColor);
        }
        t += volumeSamplingRate;
    }

    // Apply exposure
    pathColor = make_float4(make_float3(pathColor) * mainExposure, pathColor.w);

    // Combine with background color
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
    const float4 color = getVolumeContribution(ray);
    float3 result = make_float3(::optix::clamp(color, 0.f, 1.f));

    // Exposure and Fog attenuation
    const float3 hit_point = ray.origin + t_hit * ray.direction;
    const float z = optix::length(eye - hit_point);
    const float fogAttenuation = z > fogStart ? optix::clamp((z - fogStart) / fogThickness, 0.f, 1.f) : 0.f;
    result = (result * (1.f - fogAttenuation) + fogAttenuation * getEnvironmentColor());

    prd.result = result;
    prd.importance = 1.f;
    prd.depth += 1;
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
