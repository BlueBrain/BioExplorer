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

#include "DepthRenderer.ih"

#include <ospray/SDK/common/Material.ih>

inline vec3f DepthRenderer_shadeRay(const uniform DepthRenderer* uniform self,
                                    varying ScreenSample& sample)
{
    Ray ray = sample.ray;
    sample.z = inf;
    sample.alpha = 1.f;

    traceRay(self->super.model, ray);

    if (ray.geomID < 0)
        // No intersection, use environment map or background color
        return make_vec3f(1.f);

    // Z-Depth
    sample.z = ray.t;

    return make_vec3f(ray.t / self->infinity);
}

void DepthRenderer_renderSample(uniform Renderer* uniform _self,
                                void* uniform perFrameData,
                                varying ScreenSample& sample)
{
    uniform DepthRenderer* uniform self =
        (uniform DepthRenderer * uniform) _self;
    sample.ray.time = inf;
    sample.rgb = DepthRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform DepthRenderer_create(void* uniform cppE)
{
    uniform DepthRenderer* uniform self = uniform new uniform DepthRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = DepthRenderer_renderSample;
    return self;
}

export void DepthRenderer_set(void* uniform _self, const uniform int spp,
                              const uniform float infinity)
{
    uniform DepthRenderer* uniform self =
        (uniform DepthRenderer * uniform) _self;

    self->super.spp = spp;
    self->infinity = infinity;
}
