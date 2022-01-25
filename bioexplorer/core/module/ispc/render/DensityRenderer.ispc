/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include "DensityRenderer.ih"

#include "utils/BioExplorerRandomGenerator.ih"

#include <ospray/SDK/math/sampling.ih>

/**
    Composes source and destination colors according to specified alpha
   correction
    @param src Source RGBA color
    @param dst Returned destination RGBA color
    @param alphaCorrection Alpha correction used to blend colors
*/
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

inline vec3f DensityRenderer_shadeRay(
    const uniform DensityRenderer* uniform self, varying ScreenSample& sample)
{
    const Ray ray = sample.ray;

    if (sample.sampleID.z == 0)
    {
        sample.alpha = 0.f;
        sample.z = inf;
    }

    // init TEA RNG //
    uniform FrameBuffer* uniform fb = self->super.fb;
    RandomTEA rng_state;
    varying RandomTEA* const uniform rng = &rng_state;
    RandomTEA__Constructor(rng, 0x290374,
                           (fb->size.x * sample.sampleID.y) +
                               sample.sampleID.x);
    const vec2f rot = RandomTEA__getFloats(rng);

    vec4f finalColor = make_vec4f(0.f);
    bool firstVisibleVoxel = true;

    float t = self->rayStep;
    while (finalColor.w < 0.9f && t < self->farPlane)
    {
        const vec3f p = ray.org + t * ray.dir;

        int hits = 0;
        const vec3f N = ray.dir;
        const linear3f localToWorld = frame(N);

        for (uniform int i = 0; i < self->sampleCount; ++i)
        {
            const vec2f halton =
                HaltonSequence_get2D(self->sampleCount * sample.sampleID.z + i);
            const vec2f r = CranleyPattersonRotation(halton, rot);
            const vec3f local_ao_dir = cosineSampleHemisphere(r);
            const vec3f ao_dir = localToWorld * local_ao_dir;

            Ray ao_ray;
            setRay(ao_ray, p, ao_dir, 0.f, self->searchLength);
            if (isOccluded(self->super.model, ao_ray))
                ++hits;
        }

        if (hits > 0)
        {
            const float a = (float)hits / (float)self->sampleCount;

            const vec3f sampleColor = make_vec3f(a, a, 1.f - a);
            const float alpha = 1.f / (float)self->sampleCount;

            composite(make_vec4f(sampleColor, alpha), finalColor, 1.0f);

            if (firstVisibleVoxel)
                sample.z = min(t, sample.z);

            firstVisibleVoxel = false;
        }

        t += self->rayStep;
    }

    sample.alpha = finalColor.w;
    return clamp(make_vec3f(finalColor) * self->exposure, make_vec3f(0.f),
                 make_vec3f(1.f));
}

void DensityRenderer_renderSample(uniform Renderer* uniform _self,
                                  void* uniform perFrameData,
                                  varying ScreenSample& sample)
{
    uniform DensityRenderer* uniform self =
        (uniform DensityRenderer * uniform) _self;
    sample.ray.time = self->timestamp;
    sample.rgb = DensityRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform DensityRenderer_create(void* uniform cppE)
{
    uniform DensityRenderer* uniform self = uniform new uniform DensityRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = DensityRenderer_renderSample;
    return self;
}

export void DensityRenderer_set(
    void* uniform _self, void* uniform bgMaterial,
    const uniform float& timestamp, const uniform int& spp,
    const uniform float& farPlane, const uniform float& searchLength,
    const uniform float& rayStep, const uniform int32& sampleCount,
    const uniform float& exposure, const uniform float& alphaCorrection)
{
    uniform DensityRenderer* uniform self =
        (uniform DensityRenderer * uniform) _self;

    // OSPRay
    self->super.spp = spp;

    // Bio explorer
    self->bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;

    self->timestamp = timestamp;
    self->exposure = exposure;

    self->rayStep = rayStep;
    self->searchLength = searchLength;
    self->sampleCount = sampleCount;
    self->farPlane = farPlane;
    self->alphaCorrection = alphaCorrection;
}

export void DensityRenderer_setTransferFunction(void* uniform _self,
                                                void* uniform value)
{
    uniform DensityRenderer* uniform self =
        (uniform DensityRenderer * uniform) _self;
    self->transferFunction = (TransferFunction * uniform) value;
}
