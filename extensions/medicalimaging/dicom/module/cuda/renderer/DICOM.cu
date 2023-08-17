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

rtDeclareVariable(int, maxBounces, , );

const float gradientOffset = 0.001f;

static __device__ void dicomShade()
{
    float3 result = make_float3(0.f);
    if (volume_map != 0)
    {
        const float voxelValue = optix::rtTex3D<float>(volume_map, texcoord3d.x, texcoord3d.y, texcoord3d.z);
        float4 voxelColor =
            calcTransferFunctionColor(tfMinValue, tfMinValue + tfRange, voxelValue, tfColors, tfOpacities);
        const float opacity = voxelColor.w;

        const float3 hit_point = ray.origin + t_hit * ray.direction;
        float cosNL = 1.f;

        if (volumeGradientShadingEnabled)
        {
            optix::size_t2 screen = output_buffer.size();
            unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

            unsigned int num_lights = lights.size();
            float3 normal = make_float3(0.f);
            const float3 positions[6] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
            for (const auto& position : positions)
            {
                const float3 p = (position * gradientOffset) + texcoord3d;
                const float voxelValue = optix::rtTex3D<float>(volume_map, p.x, p.y, p.z);
                if (voxelValue > DEFAULT_VOLUME_SHADING_THRESHOLD)
                    normal += voxelValue * position;
            }
            normal = ::optix::normalize(-1.f * normal);
            if (length(normal) > 0.f)
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
                        lightDirection = optix::normalize(pos - hit_point);
                    }
                    else
                    {
                        lightDirection = -light.pos;
                        if (shadows > 0.f && softShadows > 0.f)
                            // Soft shadows
                            lightDirection +=
                                softShadows * make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f);
                        lightDirection = optix::normalize(lightDirection);
                    }
                    cosNL = optix::dot(normal, lightDirection);

                    // Specular
                    if (cosNL > 0.f)
                    {
                        cosNL = DEFAULT_SHADING_AMBIENT + (1.f - DEFAULT_SHADING_AMBIENT) * cosNL;
                        const float power = pow(cosNL, specularExponent);
                        voxelColor =
                            make_float4(make_float3(voxelColor) * cosNL + power * volumeSpecularColor, voxelColor.w);
                    }

                    // Ambient light
                    voxelColor = make_float4(make_float3(voxelColor) * DEFAULT_SHADING_ALPHA_RATIO, voxelColor.w);
                }
        }
        result += cosNL * make_float3(voxelColor) * opacity;

        if (opacity < 1.f)
            if (prd.depth < maxBounces)
            {
                PerRayData_radiance refracted_prd;
                refracted_prd.result = make_float3(0.f);
                refracted_prd.importance = prd.importance * (1.f - opacity);
                refracted_prd.depth = prd.depth + 1;

                const optix::Ray refracted_ray(hit_point, ray.direction, radianceRayType, sceneEpsilon);
                rtTrace(top_object, refracted_ray, refracted_prd);
                result = result * opacity + (1.f - opacity) * refracted_prd.result;
            }
            else
            {
                // We have reached the max depth for rays. Fall back to environment color for now :-/
                result = result * opacity + (1.f - opacity) * getEnvironmentColor(ray.direction);
            }
    }
    else
        result = Kd;

    // Fog attenuation
    const float3 hit_point = ray.origin + t_hit * ray.direction;
    const float z = optix::length(eye - hit_point);
    const float fogAttenuation = z > fogStart ? optix::clamp((z - fogStart) / fogThickness, 0.f, 1.f) : 0.f;
    result = (result * (1.f - fogAttenuation) + fogAttenuation * getEnvironmentColor(ray.direction));

    // Final result
    prd.result = result;
}

RT_PROGRAM void any_hit_shadow()
{
    prd.result = make_float3(0.f);
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
