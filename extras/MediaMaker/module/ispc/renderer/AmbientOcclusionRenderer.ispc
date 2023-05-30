/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include "AmbientOcclusionRenderer.ih"

#include <brayns/ispc/render/utils/AdvancedMaterial.ih>

#include <ospray/SDK/math/random.ih>
#include <ospray/SDK/math/sampling.ih>

inline vec3f AmbientOcclusionRenderer_shadeRay(const uniform AmbientOcclusionRenderer* uniform self,
                                               varying ScreenSample& sample)
{
    Ray ray = sample.ray;
    sample.z = inf;
    sample.alpha = 0.f;

    const uniform int sampleCnt = self->samplesPerFrame;
    const uniform int accumID = reduce_max(sample.sampleID.z) * self->samplesPerFrame;

    traceRay(self->super.model, ray);

    if (ray.geomID < 0)
        // No intersection, use environment map or background color
        return make_vec3f(0.f);

    sample.z = ray.t;
    sample.alpha = 1.f;

    DifferentialGeometry dg;
    postIntersect(self->super.model, dg, ray, DG_NS | DG_NG | DG_NORMALIZE | DG_FACEFORWARD);

    // init TEA RNG //
    uniform FrameBuffer* uniform fb = self->super.fb;
    RandomTEA rng_state;
    varying RandomTEA* const uniform rng = &rng_state;
    RandomTEA__Constructor(rng, 0x290374, (fb->size.x * sample.sampleID.y) + sample.sampleID.x);
    const vec2f rot = RandomTEA__getFloats(rng);

    int hits = 0;
    const vec3f N = dg.Ns;
    const linear3f localToWorld = frame(N);

    for (uniform int i = 0; i < sampleCnt; i++)
    {
        const vec2f halton = HaltonSequence_get2D(sampleCnt * accumID + i);
        const vec2f r = CranleyPattersonRotation(halton, rot);
        const vec3f local_ao_dir = cosineSampleHemisphere(r);
        vec3f ao_dir = normalize(localToWorld * local_ao_dir);
        if (dot(ao_dir, N) < 0.f)
            ao_dir = neg(ao_dir);

        Ray ao_ray;
        setRay(ao_ray, dg.P, ao_dir, dg.epsilon, self->aoRayLength);
        traceRay(self->super.model, ao_ray);
        if (ao_ray.geomID >= 0)
            hits++;
    }

    return make_vec3f(1.0f - hits / (float)sampleCnt);
}

void AmbientOcclusionRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData,
                                           varying ScreenSample& sample)
{
    uniform AmbientOcclusionRenderer* uniform self = (uniform AmbientOcclusionRenderer * uniform) _self;
    sample.ray.time = inf;
    sample.rgb = AmbientOcclusionRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform AmbientOcclusionRenderer_create(void* uniform cppE)
{
    uniform AmbientOcclusionRenderer* uniform self = uniform new uniform AmbientOcclusionRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = AmbientOcclusionRenderer_renderSample;
    return self;
}

export void AmbientOcclusionRenderer_set(void* uniform _self, const uniform int spp, const uniform int samplesPerFrame,
                                         const uniform float aoRayLength)
{
    uniform AmbientOcclusionRenderer* uniform self = (uniform AmbientOcclusionRenderer * uniform) _self;

    self->super.spp = spp;
    self->samplesPerFrame = samplesPerFrame;
    self->aoRayLength = aoRayLength;
}
