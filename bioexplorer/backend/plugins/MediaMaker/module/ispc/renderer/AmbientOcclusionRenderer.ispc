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

#include "AmbientOcclusionRenderer.ih"

#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.ih>
#include <platform/engines/ospray/ispc/render/utils/RandomGenerator.ih>
#include <platform/engines/ospray/ispc/render/utils/RendererUtils.ih>

#include <ospray/SDK/math/random.ih>
#include <ospray/SDK/math/sampling.ih>

inline vec3f AmbientOcclusionRenderer_shadeRay(const uniform AmbientOcclusionRenderer* uniform self,
                                               varying ScreenSample& sample)
{
    Ray ray = sample.ray;
    sample.z = inf;
    sample.alpha = 0.f;
    float tmax = self->aoRayLength;

    const uniform int sampleCnt = self->samplesPerFrame;
    const uniform int accumID = reduce_max(sample.sampleID.z) * self->samplesPerFrame;

    uniform FrameBuffer* uniform fb = self->super.fb;
    RandomTEA rng_state;
    varying RandomTEA* const uniform rng = &rng_state;
    RandomTEA__Constructor(rng, 0x290374, (fb->size.x * sample.sampleID.y) + sample.sampleID.x);
    const vec2f rot = RandomTEA__getFloats(rng);

    vec3f color = make_vec3f(0.f);
    float alpha = 0.f;
    uint32 depth = 0;
    float oldlocalRefraction = 1.f;
    bool moreRebounds = true;

    while (moreRebounds && depth < self->maxBounces && alpha < 1.f)
    {
        traceRay(self->super.model, ray);

        if (ray.geomID < 0)
            moreRebounds = false;
        else
        {
            DifferentialGeometry dg;
            postIntersect(self->super.model, dg, ray,
                          DG_NS | DG_NG | DG_NORMALIZE | DG_FACEFORWARD | DG_MATERIALID | DG_COLOR | DG_TEXCOORD);

            vec3f normal = dg.Ns;
            uniform AdvancedMaterial* material = (uniform AdvancedMaterial*)dg.material;

            float opacity = dg.color.w;
            float reflection = 0.f;
            float refraction = 1.f;
            if (material)
                foreach_unique(mat in material)
                {
                    opacity = mat->d;

                    // Refraction
                    refraction = mat->refraction;
                    if (valid(mat->map_d))
                        opacity = mat->d * get1f(mat->map_d, dg);

                    // Reflection
                    reflection = mat->reflection;
                    if (valid(mat->map_reflection))
                    {
                        const vec3f value = get3f(mat->map_reflection, dg);
                        reflection *= (value.x + value.y + value.z) / 3.f;
                    }

                    // Glossiness
                    if (mat->glossiness < 1.f)
                    {
                        const vec3f randomNormal = (1.f - mat->glossiness) *
                                                   getRandomVector(self->useHardwareRandomizer, self->super.fb->size.x,
                                                                   sample, normal, sample.sampleID.z);
                        normal = normalize(normal + randomNormal);
                    }
                }

            int hits = 0;
            const linear3f localToWorld = frame(normal);
            for (uniform int i = 0; i < sampleCnt; i++)
            {
                const vec2f halton = HaltonSequence_get2D(sampleCnt * accumID + i);
                const vec2f r = CranleyPattersonRotation(halton, rot);
                const vec3f local_ao_dir = cosineSampleHemisphere(r);
                vec3f ao_dir = normalize(localToWorld * local_ao_dir);
                if (dot(ao_dir, normal) < 0.f)
                    ao_dir = neg(ao_dir);

                Ray ao_ray;
                setRay(ao_ray, dg.P, ao_dir, dg.epsilon, self->aoRayLength);
                traceRay(self->super.model, ao_ray);
                if (ao_ray.geomID >= 0)
                    hits++;
            }
            const vec3f Kd = make_vec3f(hits) / (float)sampleCnt;

            // Prepare ray for next iteration
            bool doRefraction = (opacity < 1.f);
            bool doReflection = (reflection > 0.f);
            if (doRefraction && doReflection)
            {
                // Switch between refraction and reflection
                const bool reflect = sample.sampleID.z % 4 == 1;
                doRefraction = !reflect;
                doReflection = reflect;
            }

            if (doRefraction)
            {
                // Refraction
                ray.dir = refractedVector(ray.dir, normal, oldlocalRefraction, refraction);
                oldlocalRefraction = refraction;
                color = Kd * opacity + color * (1.f - opacity);
            }
            else if (doReflection)
            {
                // Reflection
                ray.dir = ray.dir - 2.f * dot(ray.dir, normal) * normal;
                color = Kd * reflection + color * (1.f - reflection);
            }
            else
            {
                color = Kd;
                moreRebounds = false;
            }

            // Prepare next ray
            ray.org = dg.P + ray.dir * dg.epsilon;
            ray.t0 = dg.epsilon;
            tmax -= ray.t;
            ray.t = inf;
            ray.primID = -1;
            ray.geomID = -1;
            ray.instID = -1;

            alpha += opacity - reflection;

            ++depth;
        }
    }

    sample.z = ray.t;
    sample.alpha = alpha;

    return make_vec3f(1.f) - color;
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
                                         const uniform float aoRayLength, const uniform int maxBounces,
                                         const uniform bool useHardwareRandomizer)
{
    uniform AmbientOcclusionRenderer* uniform self = (uniform AmbientOcclusionRenderer * uniform) _self;

    self->super.spp = spp;
    self->samplesPerFrame = samplesPerFrame;
    self->aoRayLength = aoRayLength;
    self->maxBounces = maxBounces;
    self->useHardwareRandomizer = useHardwareRandomizer;
}
