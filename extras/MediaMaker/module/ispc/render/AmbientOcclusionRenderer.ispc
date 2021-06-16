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

#include "AmbientOcclusionRenderer.ih"
#include "DefaultMaterial.ih"

#include <ospray/SDK/common/Material.ih>
#include <ospray/SDK/math/random.ih>
#include <ospray/SDK/math/sampling.ih>

inline vec3f AmbientOcclusionRenderer_shadeRay(const uniform AmbientOcclusionRenderer* uniform self, varying ScreenSample& sample)
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

    vec3f superColor = make_vec3f(1.f);

    for (uniform int i = 0; i < sampleCnt; i++)
    {
        const vec2f halton = HaltonSequence_get2D(sampleCnt * accumID + i);
        const vec2f r = CranleyPattersonRotation(halton, rot);
        const vec3f local_ao_dir = cosineSampleHemisphere(r);
        const vec3f ao_dir = localToWorld * local_ao_dir;

        Ray ao_ray;
        setRay(ao_ray, dg.P, ao_dir, 0.0f, self->aoRayLength);
        if (dot(ao_dir, N) < 0.05f || isOccluded(self->super.model, ao_ray))
            hits++;
    }

    const vec3f color = superColor * make_vec3f(1.0f - hits / (float)sampleCnt);
    sample.alpha = 1.f;

    return color;
}

void AmbientOcclusionRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData, varying ScreenSample& sample)
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
