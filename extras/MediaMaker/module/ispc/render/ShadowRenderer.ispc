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

#include "DefaultMaterial.ih"
#include "ShadowRenderer.ih"

#include <ospray/SDK/common/Material.ih>
#include <ospray/SDK/math/random.ih>
#include <ospray/SDK/math/sampling.ih>

inline vec3f ShadowRenderer_shadeRay(const uniform ShadowRenderer* uniform self,
                                     varying ScreenSample& sample)
{
    Ray ray = sample.ray;
    sample.z = inf;
    sample.alpha = 0.f;

    const uniform int sampleCnt =
        self->softness == 0.f ? 1 : self->samplesPerFrame;
    const uniform int accumID =
        reduce_max(sample.sampleID.z) * self->samplesPerFrame;

    traceRay(self->super.model, ray);

    if (ray.geomID < 0)
        // No intersection, use environment map or background color
        return make_vec3f(0.f);

    sample.z = ray.t;
    sample.alpha = 1.f;

    DifferentialGeometry dg;
    postIntersect(self->super.model, dg, ray,
                  DG_NS | DG_NG | DG_NORMALIZE | DG_FACEFORWARD);

    // init TEA RNG //
    uniform FrameBuffer* uniform fb = self->super.fb;
    RandomTEA rng_state;
    varying RandomTEA* const uniform rng = &rng_state;
    RandomTEA__Constructor(rng, 0x290374,
                           (fb->size.x * sample.sampleID.y) +
                               sample.sampleID.x);
    const vec2f rot = RandomTEA__getFloats(rng);

    int hits = 0;
    vec3f superColor = make_vec3f(1.f);

    for (uniform int l = 0; l < self->numLights; l++)
    {
        const uniform Light* uniform light = self->lights[l];

        for (uniform int i = 0; i < sampleCnt; i++)
        {
            const vec2f halton =
                HaltonSequence_get2D(sampleCnt * accumID + i + l);
            const vec2f samplingLocation =
                self->softness * (make_vec2f(0.5f) +
                                  0.5f * CranleyPattersonRotation(halton, rot));
            const varying Light_SampleRes lightSample =
                light->sample(light, dg, samplingLocation);
            const vec3f lightDirection = lightSample.dir;
            const linear3f localToWorld = frame(lightDirection);

            Ray shadow_ray;
            setRay(shadow_ray, dg.P, lightDirection, 0.0f, self->rayLength);
            if (isOccluded(self->super.model, shadow_ray))
                hits++;
        }
    }

    return superColor *
           make_vec3f(1.0f - hits / (float)(sampleCnt * self->numLights));
}

void ShadowRenderer_renderSample(uniform Renderer* uniform _self,
                                 void* uniform perFrameData,
                                 varying ScreenSample& sample)
{
    uniform ShadowRenderer* uniform self =
        (uniform ShadowRenderer * uniform) _self;
    sample.ray.time = inf;
    sample.rgb = ShadowRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform ShadowRenderer_create(void* uniform cppE)
{
    uniform ShadowRenderer* uniform self = uniform new uniform ShadowRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = ShadowRenderer_renderSample;
    return self;
}

export void ShadowRenderer_set(void* uniform _self, const uniform int spp,
                               void** uniform lights, uniform int32 numLights,
                               const uniform int samplesPerFrame,
                               const uniform float rayLength,
                               const uniform float softness)
{
    uniform ShadowRenderer* uniform self =
        (uniform ShadowRenderer * uniform) _self;

    self->lights = (const uniform Light* uniform* uniform)lights;
    self->numLights = numLights;

    self->super.spp = spp;
    self->samplesPerFrame = samplesPerFrame;
    self->rayLength = rayLength;
    self->softness = softness;
}
