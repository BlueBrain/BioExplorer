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

#include "RadianceRenderer.ih"

#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.ih>

#include <ospray/SDK/common/Material.ih>

inline vec3f RadianceRenderer_shadeRay(const uniform RadianceRenderer* uniform self, varying ScreenSample& sample)
{
    Ray ray = sample.ray;
    sample.z = inf;
    sample.alpha = 0.f;

    traceRay(self->super.model, ray);

    if (ray.geomID < 0)
        // No intersection, use environment map or background color
        return make_vec3f(0.f);

    DifferentialGeometry dg;
    postIntersect(self->super.model, dg, ray,
                  DG_NS | DG_NG | DG_NORMALIZE | DG_FACEFORWARD | DG_MATERIALID | DG_COLOR | DG_TEXCOORD);

    uniform AdvancedMaterial* material = (uniform AdvancedMaterial*)dg.material;

    vec3f radiance = make_vec3f(0.f);
    if (material)
        foreach_unique(mat in material)
        {
            radiance = make_vec3f(mat->a);
            if (valid(mat->map_a))
            {
                const vec4f radianceFromMap = get4f(mat->map_a, dg);
                radiance = make_vec3f(radianceFromMap);
            }
        }

    sample.z = ray.t;   // Z-Depth
    sample.alpha = 1.f; // Transparency is not managed by this renderer

    return radiance;
}

void RadianceRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData,
                                   varying ScreenSample& sample)
{
    uniform RadianceRenderer* uniform self = (uniform RadianceRenderer * uniform) _self;
    sample.ray.time = inf;
    sample.rgb = RadianceRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform RadianceRenderer_create(void* uniform cppE)
{
    uniform RadianceRenderer* uniform self = uniform new uniform RadianceRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = RadianceRenderer_renderSample;
    return self;
}

export void RadianceRenderer_set(void* uniform _self, const uniform int spp)
{
    uniform RadianceRenderer* uniform self = (uniform RadianceRenderer * uniform) _self;

    self->super.spp = spp;
}
