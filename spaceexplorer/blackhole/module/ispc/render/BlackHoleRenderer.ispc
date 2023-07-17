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

#include "BlackHoleRenderer.ih"

#include <platform/engines/ospray/ispc/render/utils/SkyBox.ih>

// Port from https://www.shadertoy.com/view/tsBXW3

inline float mix(float x, float y, float a)
{
    return x * (1.f - a) + y * a;
}

inline vec3f mix(vec3f x, vec3f y, float a)
{
    return x * (1.f - a) + y * a;
}

inline vec3f mix(vec3f x, vec3f y, vec3f a)
{
    return x * (1.f - a) + y * a;
}

inline vec3f mod(vec3f v, int m)
{
    return make_vec3f(v.x - m * floor(v.x / m), v.y - m * floor(v.y / m), v.z - m * floor(v.z / m));
}

inline float mod(float v, float u)
{
    return v - u * floor(v / u);
}

inline float hash(float x)
{
    return frac(sin(x) * 152754.742f);
}

inline float hash(const vec2f& x)
{
    return hash(x.x + hash(x.y));
}

inline float value(const vec2f& p, float f) // value noise
{
    float bl = hash(floor(p * f + make_vec2f(0.f, 0.f)));
    float br = hash(floor(p * f + make_vec2f(1.f, 0.f)));
    float tl = hash(floor(p * f + make_vec2f(0.f, 1.f)));
    float tr = hash(floor(p * f + make_vec2f(1.f, 1.f)));

    vec2f fr = frac(p * f);
    fr = (3.f - 2.f * fr) * fr * fr;
    float b = mix(bl, br, fr.x);
    float t = mix(tl, tr, fr.x);
    return mix(b, t, fr.y);
}

vec4f raymarchDisk(const uniform BlackHoleRenderer* uniform self, const vec3f& dir, const vec3f& zeroPos)
{
    vec3f position = zeroPos;
    float lengthPos = length(make_vec3f(position.x, position.z, 0.f));
    float dist = min(1.f, lengthPos * (1.f / self->blackHoleSize) * 0.5f) * self->blackHoleSize * 0.4f * (1.f / self->diskTextureLayers) /
                 abs(dir.y);

    position = position + dist * self->diskTextureLayers * dir * 0.5f;

    vec3f deltaPos = make_vec3f(0.f);
    deltaPos.x = -zeroPos.z * 0.01f + zeroPos.x;
    deltaPos.y = zeroPos.x * 0.01f + zeroPos.z;
    deltaPos = normalize(deltaPos - make_vec3f(zeroPos.x, zeroPos.z, 0.f));

    float parallel = dot(make_vec3f(dir.x, dir.z, 0.f), deltaPos);
    parallel = parallel / sqrt(lengthPos);
    parallel = parallel * 0.5f;

    float redShift = parallel + 0.3f;
    redShift = redShift * redShift;
    redShift = clamp(redShift, 0.f, 1.f);

    float disMix = clamp((lengthPos - self->blackHoleSize * 2.f) * (1.f / self->blackHoleSize) * 0.24f, 0.f, 1.f);
    vec3f insideCol = mix(make_vec3f(1.f, 0.8f, 0.f), make_vec3f(0.5f, 0.13f, 0.02f) * 0.2f, disMix);

    insideCol = insideCol * mix(make_vec3f(0.4f, 0.2f, 0.1f), make_vec3f(1.6f, 2.4f, 4.f), redShift);
    insideCol = insideCol * 1.25f;
    redShift += 0.12f;
    redShift *= redShift;

    vec4f o = make_vec4f(0.f);

    for (float i = 0.f; i < self->diskTextureLayers; i++)
    {
        position = position - dist * dir;

        float intensity = clamp(1.f - abs((i - 0.8f) * (1.f / self->diskTextureLayers) * 2.f), 0.f, 1.f);
        float lengthPos = length(make_vec3f(position.x, position.z, 0.f));
        float distMult = 1.;

        distMult *= clamp((lengthPos - self->blackHoleSize * 0.75f) * (1.f / self->blackHoleSize) * 1.5f, 0.f, 1.f);
        distMult *= clamp((self->blackHoleSize * 10.f - lengthPos) * (1.f / self->blackHoleSize) * 0.2f, 0.f, 1.f);
        distMult *= distMult;

        float u = lengthPos + self->super.timestamp * self->blackHoleSize * 0.3f + intensity * self->blackHoleSize * 0.2f;

        vec2f xy;
        float rot = mod(self->super.timestamp * self->diskRotationSpeed, 8192.f);
        xy.x = -position.z * sin(rot) + position.x * cos(rot);
        xy.y = position.x * sin(rot) + position.z * cos(rot);

        const float x = abs(xy.x / xy.y);
        const float angle = 0.02f * atan(x);

        const float f = 70.f;
        float noise = value(make_vec2f(angle, u * (1.f / self->blackHoleSize) * 0.05f), f);
        noise = noise * 0.66f + 0.33f * value(make_vec2f(angle, u * (1.f / self->blackHoleSize) * 0.05f), f * 2.f);

        const float extraWidth = noise * 1.f * (1.f - clamp(i * (1.f / self->diskTextureLayers) * 2.f - 1.f, 0.f, 1.f));
        const float alpha =
            clamp(noise * (intensity + extraWidth) * ((1.f / self->blackHoleSize) * 10.f + 0.01f) * dist * distMult, 0.f, 1.f);
        const vec3f col = 2.f * mix(make_vec3f(0.3f, 0.2f, 0.15f) * insideCol, insideCol, min(1.f, intensity * 2.f));

        const vec3f t = col * alpha + make_vec3f(o) * (1.f - alpha);
        o = make_vec4f(clamp(t, make_vec3f(0.f), make_vec3f(1.f)), o.w * (1.f - alpha) + alpha);

        lengthPos *= 1.f / self->blackHoleSize;

        o = o + make_vec4f(make_vec3f(redShift * (intensity * 1.f + 0.5f) * (1.f / self->diskTextureLayers) * 100.f * distMult /
                                      (lengthPos * lengthPos)),
                           0.f);
    }

    o = make_vec4f(clamp(make_vec3f(o) - 0.005f, make_vec3f(0.f), make_vec3f(1.f)), o.w);
    return o;
}

inline void composite(const varying vec4f& src, varying vec4f& dst, const varying float alphaCorrection)
{
    const float alpha = 1.f - pow(1.f - min(src.w, 1.f - 1.f / 256.f), alphaCorrection);
    const float a = alpha * (1.f - dst.w);
    dst.x = dst.x + src.x * a;
    dst.y = dst.y + src.y * a;
    dst.z = dst.z + src.z * a;
    dst.w += (alpha * (1.f - dst.w));
}

inline vec3f BlackHoleRenderer_shadeRay(const uniform BlackHoleRenderer* uniform self, varying ScreenSample& sample)
{
    vec4f colOut = make_vec4f(0.f);
    vec4f outCol = make_vec4f(0.f);
    vec4f glow = make_vec4f(0.f);
    vec4f col = make_vec4f(0.f);

    if (sample.sampleID.z == 0)
    {
        sample.alpha = 0.f;
        sample.z = inf;
    }

    vec3f pos = sample.ray.org;
    vec3f dir = sample.ray.dir;

    for (int disks = 0; disks < self->nbDisks; ++disks) // steps
    {
        for (int h = 0; h < 6; h++)                     // reduces tests for exit conditions (to minimise branching)
        {
            float dotpos = dot(pos, pos);
            float invDist = sqrt(1.f / dotpos);           // 1 / distance to black hole
            float centDist = dotpos * invDist;            // distance to black hole
            float stepDist = 0.92 * abs(pos.y / (dir.y)); // conservative distance to disk (y==0)
            float farLimit = centDist * 0.5f;             // limit step size far from to black hole
            float closeLimit = centDist * 0.1f + 0.05f * centDist * centDist * (1.f / self->blackHoleSize); // limit step size closse to BH
            stepDist = min(stepDist, min(farLimit, closeLimit));

            float invDistSqr = invDist * invDist;
            float bendForce = stepDist * invDistSqr * self->blackHoleSize * 0.625f; // bending force
            dir = normalize(dir - (bendForce * invDist) * pos);                     // bend ray towards BH
            pos = pos + stepDist * dir;

            glow = glow + make_vec4f(1.2f, 1.1f, 1.f, 1.f) * (0.01f * stepDist * invDistSqr * invDistSqr *
                                                              clamp(centDist * 2.f - 1.2f, 0.f, 1.f)); // adds fairly cheap glow
        }

        float dist2 = length(pos);

        if (dist2 < self->blackHoleSize * 0.1f) // ray sucked in to BH
        {
            outCol = make_vec4f(make_vec3f(col) * col.w + make_vec3f(glow) * (1. - col.w), 1.);
            break;
        }

        else if (dist2 > self->blackHoleSize * 1000.f) // ray escaped BH
        {
            vec4f bg;
            if (self->grid)
            {
                bg =
                    make_vec4f((int)((pos.x + 1000.f) * 0.01f) % 2 == 0 ? 1.f : 0.5f, (int)((pos.y + 1000.f) * 0.01f) % 2 == 0 ? 1.f : 0.5f,
                               (int)((pos.z + 1000.f) * 0.01f) % 2 == 0 ? 1.f : 0.f, 0.5f);
            }
            else
            {
                Ray envRay = sample.ray;

                envRay.org = pos;
                envRay.dir = dir;
                bg = skyboxMapping((Renderer*)self, envRay, self->super.bgMaterial);
            }
            outCol = make_vec4f(make_vec3f(col) * col.w + make_vec3f(bg) * (1. - col.w) + make_vec3f(glow) * (1. - col.w), 1.);
            break;
        }

        else if (abs(pos.y) <= self->blackHoleSize * 0.002f) // ray hit accretion disk
        {
            vec4f diskCol = raymarchDisk(self, dir, pos);    // render disk
            pos.y = 0.f;
            pos = pos + abs(self->blackHoleSize * 0.001f / dir.y) * dir;
            col = make_vec4f(make_vec3f(diskCol) * (1.f - col.w) + make_vec3f(col), col.w + diskCol.w * (1.f - col.w));
        }
    }

    // if the ray never escaped or got sucked in
    if (outCol.x == 100.f)
        outCol = make_vec4f(make_vec3f(col) + make_vec3f(glow) * (col.w + glow.w), 1.f);

    return clamp(make_vec3f(outCol * self->exposure), make_vec3f(0.f), make_vec3f(1.f));
}

void BlackHoleRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData, varying ScreenSample& sample)
{
    uniform BlackHoleRenderer* uniform self = (uniform BlackHoleRenderer * uniform) _self;
    sample.ray.time = self->super.timestamp;
    sample.rgb = BlackHoleRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform BlackHoleRenderer_create(void* uniform cppE)
{
    uniform BlackHoleRenderer* uniform self = uniform new uniform BlackHoleRenderer;
    Renderer_Constructor(&self->super.super, cppE);
    self->super.super.renderSample = BlackHoleRenderer_renderSample;
    return self;
}

export void BlackHoleRenderer_set(void* uniform _self, void* uniform bgMaterial, const uniform float timestamp, const uniform int spp,
                                  const uniform float exposure, const uniform int nbDisks, const uniform bool grid,
                                  const uniform float diskRotationSpeed, const uniform uint32 diskTextureLayers,
                                  const uniform float blackHoleSize)
{
    uniform BlackHoleRenderer* uniform self = (uniform BlackHoleRenderer * uniform) _self;

    self->super.bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;

    self->super.super.spp = spp;
    self->super.timestamp = timestamp;

    self->exposure = exposure;
    self->nbDisks = nbDisks;
    self->grid = grid;
    self->diskRotationSpeed = diskRotationSpeed;
    self->diskTextureLayers = diskTextureLayers;
    self->blackHoleSize = blackHoleSize;
}
