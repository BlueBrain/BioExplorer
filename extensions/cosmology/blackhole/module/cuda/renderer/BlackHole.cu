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

#include <optix_world.h>

#include <platform/engines/optix6/cuda/Context.cuh>
#include <platform/engines/optix6/cuda/Environment.cuh>
#include <platform/engines/optix6/cuda/Helpers.cuh>
#include <platform/engines/optix6/cuda/Random.cuh>

// Rendering
rtDeclareVariable(uint, displayGrid, , );
rtDeclareVariable(int, nbDisks, , );
rtDeclareVariable(float, diskRotationSpeed, , );
rtDeclareVariable(int, diskTextureLayers, , );
rtDeclareVariable(float, size, , );
rtDeclareVariable(float, timestamp, , );

// Port from https://www.shadertoy.com/view/tsBXW3

static __device__ inline float2 frac2(const float2 x)
{
    return x - optix::floor(x);
}

static __device__ inline float3 mix(float3 x, float3 y, float a)
{
    return x * (1.f - a) + y * a;
}

static __device__ inline float mod(float v, float u)
{
    return v - u * floor(v / u);
}

static __device__ inline float value(const float2& p, float f)
{
    float bl = hash(floor(p * f + make_float2(0.f, 0.f)));
    float br = hash(floor(p * f + make_float2(1.f, 0.f)));
    float tl = hash(floor(p * f + make_float2(0.f, 1.f)));
    float tr = hash(floor(p * f + make_float2(1.f, 1.f)));

    float2 fr = frac2(p * f);
    fr = (3.f - 2.f * fr) * fr * fr;
    float b = mix(bl, br, fr.x);
    float t = mix(tl, tr, fr.x);
    return mix(b, t, fr.y);
}

static __device__ inline float4 raymarchDisk(const float3& dir, const float3& zeroPos)
{
    float3 position = zeroPos;
    float lengthPos = ::optix::length(make_float3(position.x, position.z, 0.f));
    float dist = min(1.f, lengthPos * (1.f / size) * 0.5f) * size * 0.4f * (1.f / diskTextureLayers) / abs(dir.y);

    position = position + dist * diskTextureLayers * dir * 0.5f;

    float3 deltaPos = make_float3(0.f);
    deltaPos.x = -zeroPos.z * 0.01f + zeroPos.x;
    deltaPos.y = zeroPos.x * 0.01f + zeroPos.z;
    deltaPos = ::optix::normalize(deltaPos - make_float3(zeroPos.x, zeroPos.z, 0.f));

    float parallel = ::optix::dot(make_float3(dir.x, dir.z, 0.f), deltaPos);
    parallel = parallel / sqrt(lengthPos);
    parallel = parallel * 0.5f;

    float redShift = parallel + 0.3f;
    redShift = redShift * redShift;
    redShift = ::optix::clamp(redShift, 0.f, 1.f);

    float disMix = ::optix::clamp((lengthPos - size * 2.f) * (1.f / size) * 0.24f, 0.f, 1.f);
    float3 insideCol = mix(make_float3(1.f, 0.8f, 0.f), make_float3(0.5f, 0.13f, 0.02f) * 0.2f, disMix);

    insideCol = insideCol * mix(make_float3(0.4f, 0.2f, 0.1f), make_float3(1.6f, 2.4f, 4.f), redShift);
    insideCol = insideCol * 1.25f;
    redShift += 0.12f;
    redShift *= redShift;

    float4 o = make_float4(0.f);

    for (float i = 0.f; i < diskTextureLayers; i++)
    {
        position = position - dist * dir;

        float intensity = ::optix::clamp(1.f - abs((i - 0.8f) * (1.f / diskTextureLayers) * 2.f), 0.f, 1.f);
        float lengthPos = ::optix::length(make_float3(position.x, position.z, 0.f));
        float distMult = 1.;

        distMult *= ::optix::clamp((lengthPos - size * 0.75f) * (1.f / size) * 1.5f, 0.f, 1.f);
        distMult *= ::optix::clamp((size * 10.f - lengthPos) * (1.f / size) * 0.2f, 0.f, 1.f);
        distMult *= distMult;

        float u = lengthPos + timestamp * size * 0.3f + intensity * size * 0.2f;

        float2 xy;
        float rot = mod(timestamp * diskRotationSpeed, 8192.f);
        xy.x = -position.z * sin(rot) + position.x * cos(rot);
        xy.y = position.x * sin(rot) + position.z * cos(rot);

        const float x = abs(xy.x / xy.y);
        const float angle = 0.02f * atan(x);

        const float f = 70.f;
        float noise = value(make_float2(angle, u * (1.f / size) * 0.05f), f);
        noise = noise * 0.66f + 0.33f * value(make_float2(angle, u * (1.f / size) * 0.05f), f * 2.f);

        const float extraWidth =
            noise * 1.f * (1.f - ::optix::clamp(i * (1.f / diskTextureLayers) * 2.f - 1.f, 0.f, 1.f));
        const float alpha =
            ::optix::clamp(noise * (intensity + extraWidth) * ((1.f / size) * 10.f + 0.01f) * dist * distMult, 0.f,
                           1.f);
        const float3 col = 2.f * mix(make_float3(0.3f, 0.2f, 0.15f) * insideCol, insideCol, min(1.f, intensity * 2.f));

        const float3 t = col * alpha + make_float3(o) * (1.f - alpha);
        o = make_float4(::optix::clamp(t, make_float3(0.f), make_float3(1.f)), o.w * (1.f - alpha) + alpha);

        lengthPos *= 1.f / size;

        o = o + make_float4(make_float3(redShift * (intensity * 1.f + 0.5f) * (1.f / diskTextureLayers) * 100.f *
                                        distMult / (lengthPos * lengthPos)),
                            0.f);
    }

    o = make_float4(::optix::clamp(make_float3(o) - 0.005f, make_float3(0.f), make_float3(1.f)), o.w);
    return o;
}

static __device__ inline void shade()
{
    float4 colOut = make_float4(0.f);
    float4 outCol = make_float4(0.f);
    float4 glow = make_float4(0.f);
    float4 col = make_float4(0.f);

    float3 pos = ray.origin;
    float3 dir = ray.direction;

    for (int disks = 0; disks < nbDisks; ++disks) // steps
    {
        for (int h = 0; h < 6; h++) // reduces tests for exit conditions (to minimise branching)
        {
            float dotpos = ::optix::dot(pos, pos);
            float invDist = sqrt(1.f / dotpos);           // 1 / distance to black hole
            float centDist = dotpos * invDist;            // distance to black hole
            float stepDist = 0.92 * abs(pos.y / (dir.y)); // conservative distance to disk (y==0)
            float farLimit = centDist * 0.5f;             // limit step size far from to black hole
            float closeLimit =
                centDist * 0.1f + 0.05f * centDist * centDist * (1.f / size); // limit step size closse to BH
            stepDist = min(stepDist, min(farLimit, closeLimit));

            float invDistSqr = invDist * invDist;
            float bendForce = stepDist * invDistSqr * size * 0.625f;     // bending force
            dir = ::optix::normalize(dir - (bendForce * invDist) * pos); // bend ray towards BH
            pos = pos + stepDist * dir;

            glow = glow + make_float4(1.2f, 1.1f, 1.f, 1.f) *
                              (0.01f * stepDist * invDistSqr * invDistSqr *
                               ::optix::clamp(centDist * 2.f - 1.2f, 0.f, 1.f)); // adds fairly cheap glow
        }

        float dist2 = ::optix::length(pos);

        if (dist2 < size * 0.1f) // ray sucked in to BH
        {
            outCol = make_float4(make_float3(col) * col.w + make_float3(glow) * (1. - col.w), 1.);
            break;
        }

        else if (dist2 > size * 1000.f) // ray escaped BH
        {
            float4 bg;
            if (displayGrid)
                bg = make_float4((int)((pos.x + 1000.f) * 0.01f) % 2 == 0 ? 1.f : 0.5f,
                                 (int)((pos.y + 1000.f) * 0.01f) % 2 == 0 ? 1.f : 0.5f,
                                 (int)((pos.z + 1000.f) * 0.01f) % 2 == 0 ? 1.f : 0.f, 0.5f);
            else
                bg = make_float4(getEnvironmentColor(dir), 0.f);

            outCol = make_float4(make_float3(col) * col.w + make_float3(bg) * (1. - col.w) +
                                     make_float3(glow) * (1. - col.w),
                                 1.);
            break;
        }

        else if (abs(pos.y) <= size * 0.002f) // ray hit accretion disk
        {
            float4 diskCol = raymarchDisk(dir, pos); // render disk
            pos.y = 0.f;
            pos = pos + abs(size * 0.001f / dir.y) * dir;
            col =
                make_float4(make_float3(diskCol) * (1.f - col.w) + make_float3(col), col.w + diskCol.w * (1.f - col.w));
        }
    }

    // if the ray never escaped or got sucked in
    if (outCol.x == 100.f)
        outCol = make_float4(make_float3(col) + make_float3(glow) * (col.w + glow.w), 1.f);

    prd.result = ::optix::clamp(outCol, 0.f, 1.f);
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
