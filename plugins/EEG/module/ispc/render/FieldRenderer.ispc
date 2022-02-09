/* Copyright (c) 2015-2022, EPFL/Blue Brain Project
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

#include "FieldRenderer.ih"
#include "algorithms/eeg.ih"

inline void composite(const varying vec4f& src, varying vec4f& dst,
                      const varying float alphaCorrection)
{
    const float alpha =
        1.f - pow(1.f - min(src.w, 1.f - 1.f / 256.f), alphaCorrection);
    const float a = alpha * (1.f - dst.w);
    dst.x = dst.x + src.x * a;
    dst.y = dst.y + src.y * a;
    dst.z = dst.z + src.z * a;
    dst.w += (alpha * (1.f - dst.w));
}

inline vec3f FieldRenderer_shadeRay(const uniform FieldRenderer* uniform self,
                                    varying ScreenSample& sample)
{
    vec4f finalColor = make_vec4f(0.f);

    vec3f dir = sample.ray.dir;

    if (sample.sampleID.z == 0)
    {
        sample.alpha = 0.f;
        sample.z = inf;
    }

    float t0, t1;
    const uniform box3f box =
        make_box3f(make_vec3f(self->userData[0] - self->cutoff,
                              self->userData[1] - self->cutoff,
                              self->userData[2] - self->cutoff),
                   make_vec3f(self->userData[3] + self->cutoff,
                              self->userData[4] + self->cutoff,
                              self->userData[5] + self->cutoff));
    const vec3f s = make_vec3f(
        (self->userData[3] - self->userData[0] + 2.f * self->cutoff) / 2.f,
        (self->userData[4] - self->userData[1] + 2.f * self->cutoff) / 2.f,
        (self->userData[5] - self->userData[2] + 2.f * self->cutoff) / 2.f);

    intersectBox(sample.ray, box, t0, t1);
    t0 = max(0.001f, t0);

#if 0
    traceRay(self->super.model, sample.ray);
    if (sample.ray.geomID >= 0)
        t1 = min(t1, sample.ray.t);
#endif

    const float diag = max(max(s.x, s.y), s.z);
    const float step = max(0.001f, diag / self->nbRaySteps);

    float t = t0 + (sample.sampleID.z % self->nbRefinementSteps) * step /
                       self->nbRefinementSteps;

    bool firstVisibleVoxel = true;

    vec3f p = sample.ray.org + t * dir;
    while (finalColor.w < 0.999f && t < t1 && self->userDataSize > 0)
    {
        float fieldValue;
        vec3f fieldDirection;
        getEEGField(p, self->super.model, self->userData, self->userDataSize,
                    self->processGeometry, self->cutoff, self->normalized,
                    fieldValue, fieldDirection);

        const float sampleOpacity =
            self->transferFunction->getOpacityForValue(self->transferFunction,
                                                       fieldValue);
        if (self->renderDirections && fieldValue > 0.f)
        {
            const vec4f sampleColor =
                make_vec4f(0.5f + 0.5f * normalize(fieldDirection),
                           sampleOpacity);
            composite(sampleColor, finalColor, self->alphaCorrection);
        }
        else
        {
            if (sampleOpacity > 0.f)
            {
                if (firstVisibleVoxel)
                {
                    firstVisibleVoxel = false;
                    sample.z = t;
                }

                const vec4f sampleColor =
                    make_vec4f(self->transferFunction->getColorForValue(
                                   self->transferFunction, fieldValue),
                               sampleOpacity);
                composite(sampleColor, finalColor, self->alphaCorrection);
                sample.alpha = finalColor.w;
            }
        }

        dir = normalize(dir +
                        self->spaceDistortion * fieldValue * fieldDirection);
        p = p + step * dir;

        t += step;
    }

    if (self->spaceDistortion > 0.f)
    {
        const vec4f gridColor =
            make_vec4f((int)((p.x + 1000.f) * 0.01f) % 2 == 0 ? 1.f : 0.5f,
                       (int)((p.y + 1000.f) * 0.01f) % 2 == 0 ? 1.f : 0.5f,
                       (int)((p.z + 1000.f) * 0.01f) % 2 == 0 ? 1.f : 0.f,
                       0.5f);
        composite(gridColor, finalColor, 0.5f);
    }

    return clamp(make_vec3f(finalColor) * self->exposure, make_vec3f(0.f),
                 make_vec3f(1.f));
}

void FieldRenderer_renderSample(uniform Renderer* uniform _self,
                                void* uniform perFrameData,
                                varying ScreenSample& sample)
{
    uniform FieldRenderer* uniform self =
        (uniform FieldRenderer * uniform) _self;
    sample.ray.time = 0.f;
    sample.rgb = FieldRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform FieldRenderer_create(void* uniform cppE)
{
    uniform FieldRenderer* uniform self = uniform new uniform FieldRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = FieldRenderer_renderSample;
    return self;
}

export void FieldRenderer_set(
    void* uniform _self, uniform float* uniform userData,
    const uniform uint64 userDataSize, const uniform int& randomNumber,
    const uniform float& timestamp, const uniform int& spp,
    void** uniform lights, const uniform int32 numLights,
    const uniform int32& nbRaySteps, const uniform int32& nbRefinementSteps,
    const uniform float& exposure, const uniform float& cutoff,
    const uniform float& alphaCorrection, const uniform bool renderDirections,
    const uniform float& spaceDistortion, const uniform bool processGeometry,
    const uniform bool normalized)
{
    uniform FieldRenderer* uniform self =
        (uniform FieldRenderer * uniform) _self;

    // OSPRay
    self->super.spp = spp;

    self->lights = (const uniform Light* uniform* uniform)lights;
    self->numLights = numLights;

    self->exposure = exposure;
    self->nbRaySteps = nbRaySteps;
    self->nbRefinementSteps = nbRefinementSteps;
    self->alphaCorrection = alphaCorrection;

    self->cutoff = cutoff;
    self->spaceDistortion = spaceDistortion;
    self->renderDirections = renderDirections;
    self->processGeometry = processGeometry;
    self->normalized = normalized;

    self->userData = userData;
    self->userDataSize = userDataSize;
}

export void FieldRenderer_setTransferFunction(void* uniform _self,
                                              void* uniform value)
{
    uniform FieldRenderer* uniform self =
        (uniform FieldRenderer * uniform) _self;
    self->transferFunction = (TransferFunction * uniform) value;
}
