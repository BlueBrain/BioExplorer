/* Copyright (c) 2019, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#pragma once

#include "../../CommonStructs.h"
#include "../Helpers.h"
#include "../Random.h"
#include <optix_world.h>

struct PerRayData_shadow
{
    float3 attenuation;
};

// Scene
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(PerRayData_shadow, prd_shadow, rtPayload, );
rtDeclareVariable(unsigned int, frame, , );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(unsigned int, radianceRayType, , );
rtDeclareVariable(unsigned int, shadowRayType, , );
rtDeclareVariable(float, epsilonFactor, , );
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );
rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, bgColor, , );
rtDeclareVariable(float4, jitter4, , );

// Lights
rtBuffer<BasicLight> lights;
rtDeclareVariable(float3, ambientLightColor, , );

// Rendering
rtDeclareVariable(int, maxBounces, , );
rtDeclareVariable(float, shadows, , );
rtDeclareVariable(float, softShadows, , );
rtDeclareVariable(int, softShadowsSamples, , );
rtDeclareVariable(float, exposure, , );
rtDeclareVariable(float, giDistance, , );
rtDeclareVariable(float, giWeight, , );
rtDeclareVariable(int, giSamples, , );
rtDeclareVariable(unsigned int, matrixFilter, , );
rtDeclareVariable(float, fogStart, , );
rtDeclareVariable(float, fogThickness, , );

rtBuffer<uchar4, 2> output_buffer;

const uint NB_MAX_SAMPLES_PER_RAY = 32;

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
                                  float p_reflectionIndex, float p_refractionIndex, float p_phong_exp,
                                  float p_glossiness, float3 p_normal)
{
    float3 color = make_float3(0.f);

    // Randomness
    optix::size_t2 screen = output_buffer.size();
    unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame);

    // Glossiness
    if (p_glossiness < 1.f)
        p_normal = optix::normalize(p_normal + (1.f - p_glossiness) *
                                                   make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));
    float light_attenuation = 1.f;
    const float3 hit_point = ray.origin + t_hit * ray.direction;

    // Surface
    // compute direct lighting
    unsigned int num_lights = lights.size();
    for (int i = 0; i < num_lights; ++i)
    {
        BasicLight light = lights[i];
        float3 lightDirection;

        float3 directLightingColor = make_float3(0.f);
        unsigned int nbSamples = (softShadows > 0.f ? softShadowsSamples : 1);
        for (int j = 0; j < nbSamples; ++j)
        {
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
            float nDl = optix::dot(p_normal, lightDirection);

            // Shadows
            if (shadows > 0.f)
            {
                if (nDl > 0.f && light.casts_shadow)
                {
                    PerRayData_shadow shadow_prd;
                    shadow_prd.attenuation = make_float3(1.f);
                    optix::Ray shadow_ray(hit_point, lightDirection, shadowRayType, epsilonFactor, giDistance);
                    rtTrace(top_shadower, shadow_ray, shadow_prd);
                    // light_attenuation is zero if completely shadowed
                    light_attenuation -= 1.f - ::optix::luminance(shadow_prd.attenuation);
                }
            }

            // If not completely shadowed, light the hit point
            if (light_attenuation > 0.f)
            {
                // Diffuse
                const float3 Lc = light.color * light_attenuation;
                directLightingColor += light_attenuation * p_Kd * nDl * Lc;

                float3 H = optix::normalize(lightDirection - ray.direction);
                float nDh = optix::dot(p_normal, H);
                if (nDh > 0.f)
                {
                    // Specular
                    float power = pow(nDh, p_phong_exp);
                    directLightingColor += p_Ks * power * Lc;
                }
            }
        }
        directLightingColor /= nbSamples;
        color += directLightingColor;
    }

    // Reflection
    if (p_reflectionIndex > 0.f)
    {
        if (prd_radiance.depth < maxBounces)
        {
            PerRayData_radiance reflected_prd;
            reflected_prd.depth = prd_radiance.depth + 1;

            const float3 R = optix::reflect(ray.direction, p_normal);
            const optix::Ray reflected_ray(hit_point, R, radianceRayType, epsilonFactor, giDistance);
            rtTrace(top_object, reflected_ray, reflected_prd);
            color = color * (1.f - p_reflectionIndex) + p_reflectionIndex * reflected_prd.result;
        }
    }

    // Refraction
    if (fmaxf(p_Ko) < 1.f && prd_radiance.depth < maxBounces)
    {
        if (prd_radiance.depth < maxBounces)
        {
            PerRayData_radiance refracted_prd;
            refracted_prd.depth = prd_radiance.depth + 1;

            const float3 R = refractedVector(ray.direction, p_normal, p_refractionIndex, 1.f);
            const optix::Ray refracted_ray(hit_point, R, radianceRayType, epsilonFactor, giDistance);
            rtTrace(top_object, refracted_ray, refracted_prd);
            color = color * p_Ko + (1.f - p_Ko) * refracted_prd.result;
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
            if (optix::dot(aa_normal, p_normal) < 0.f)
                aa_normal = -aa_normal;

            const optix::Ray aa_ray(hit_point, aa_normal, shadowRayType, epsilonFactor, giDistance);
            rtTrace(top_object, aa_ray, aa_prd);
            aa_color = aa_color + giWeight * aa_prd.result;
        }
        color += aa_color / giSamples;
    }

    // Only opaque surfaces are affected by Global Illumination
    if (fmaxf(p_Ko) == 1.f && prd_radiance.depth < maxBounces)
    {
        // Color bleeding
        if (giWeight > 0.f && prd_radiance.depth == 0)
        {
            PerRayData_radiance new_prd;
            new_prd.depth = prd_radiance.depth + 1;

            float3 ra_normal = ::optix::normalize(make_float3(rnd(seed) - 0.5f, rnd(seed) - 0.5f, rnd(seed) - 0.5f));
            if (optix::dot(ra_normal, p_normal) < 0.f)
                ra_normal = -ra_normal;

            const float3 origin = hit_point + epsilonFactor * ra_normal;
            const optix::Ray ra_ray = optix::make_Ray(origin, ra_normal, radianceRayType, epsilonFactor, ray.tmax);
            rtTrace(top_shadower, ra_ray, new_prd);
            color += giWeight * new_prd.result;
        }
    }

    // Matrix filter :)
    if (matrixFilter)
        color = make_float3(color.x * 0.666f, color.y * 0.8f, color.z * 0.666f);

    // Fog attenuation
    const float z = optix::length(eye - hit_point);
    const float fogAttenuation = z > fogStart ? optix::clamp((z - fogStart) / fogThickness, 0.f, 1.f) : 0.f;
    color = exposure * (color * (1.f - fogAttenuation) + fogAttenuation * bgColor);

    prd_radiance.result = color;
}
