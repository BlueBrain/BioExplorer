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

#include "AlbedoRenderer.ih"

#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.ih>
#include <platform/engines/ospray/ispc/render/utils/RandomGenerator.ih>
#include <platform/engines/ospray/ispc/render/utils/RendererUtils.ih>

#include <ospray/SDK/common/Material.ih>

inline vec3f AlbedoRenderer_shadeRay(const uniform AlbedoRenderer* uniform self, varying ScreenSample& sample)
{
    Ray ray = sample.ray;
    sample.z = inf;
    sample.alpha = 0.f;

    vec3f color = make_vec3f(0.f);
    float alpha = 0.f;
    uint32 depth = 0;
    float oldlocalRefraction = 1.f;
    bool moreRebounds = true;

    while (moreRebounds && depth < self->super.maxBounces && alpha < 1.f)
    {
        traceRay(self->super.super.super.model, ray);

        if (ray.geomID < 0)
            moreRebounds = false;
        else
        {
            DifferentialGeometry dg;
            postIntersect(self->super.super.super.model, dg, ray,
                          DG_NS | DG_NG | DG_NORMALIZE | DG_FACEFORWARD | DG_MATERIALID | DG_COLOR | DG_TEXCOORD);

            vec3f normal = dg.Ng;
            uniform AdvancedMaterial* material = (uniform AdvancedMaterial*)dg.material;

            float opacity = dg.color.w;
            float reflection = 0.f;
            float refraction = 1.f;
            vec3f Kd = make_vec3f(dg.color);
            if (material)
                foreach_unique(mat in material)
                {
                    // Opacity
                    opacity = mat->d;

                    // Diffuse color
                    Kd = mat->Kd * make_vec3f(dg.color);
                    if (valid(mat->map_Kd))
                    {
                        const vec4f diffuseColorFromMap = get4f(mat->map_Kd, dg);
                        Kd = make_vec3f(dg.color * diffuseColorFromMap);
                    }

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
                        const vec3f randomNormal =
                            (1.f - mat->glossiness) * getRandomVector(self->super.useHardwareRandomizer,
                                                                      self->super.super.super.fb->size.x, sample,
                                                                      normal, sample.sampleID.z);
                        normal = normalize(normal + randomNormal);
                    }

                    // User data
                    if (mat->castUserData && self->super.simulationData)
                    {
                        const vec4f simulationColor = getSimulationValue(&self->super, &dg, ray.primID);
                        Kd = make_vec3f(simulationColor);
                    }
                }

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
            ray.t0 = 0.f;
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

    return color;
}

void AlbedoRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData,
                                 varying ScreenSample& sample)
{
    uniform AlbedoRenderer* uniform self = (uniform AlbedoRenderer * uniform) _self;
    sample.ray.time = inf;
    sample.rgb = AlbedoRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform AlbedoRenderer_create(void* uniform cppE)
{
    uniform AlbedoRenderer* uniform self = uniform new uniform AlbedoRenderer;
    Renderer_Constructor(&self->super.super.super, cppE);
    self->super.super.super.renderSample = AlbedoRenderer_renderSample;
    return self;
}

export void AlbedoRenderer_set(void* uniform _self, const uniform int spp, const uniform int maxBounces,
                               const uniform bool useHardwareRandomizer, uniform float* uniform simulationData,
                               const uniform uint64 simulationDataSize)
{
    uniform AlbedoRenderer* uniform self = (uniform AlbedoRenderer * uniform) _self;

    self->super.super.super.spp = spp;
    self->super.useHardwareRandomizer = useHardwareRandomizer;
    self->super.maxBounces = maxBounces;
    self->super.simulationData = (uniform float* uniform)simulationData;
    self->super.simulationDataSize = simulationDataSize;
}
