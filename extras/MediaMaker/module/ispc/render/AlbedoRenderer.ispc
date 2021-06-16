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

#include "AlbedoRenderer.ih"
#include "DefaultMaterial.ih"

#include <ospray/SDK/common/Material.ih>

inline vec3f AlbedoRenderer_shadeRay(const uniform AlbedoRenderer* uniform self, varying ScreenSample& sample)
{
    Ray ray = sample.ray;
    sample.z = inf;
    sample.alpha = 0.f;

    traceRay(self->super.model, ray);

    if (ray.geomID < 0)
        // No intersection, use environment map or background color
        return make_vec3f(0.f);

    DifferentialGeometry dg;
    postIntersect(self->super.model, dg, ray, DG_NS | DG_NG | DG_NORMALIZE | DG_FACEFORWARD | DG_MATERIALID | DG_COLOR | DG_TEXCOORD);

    uniform DefaultMaterial* material = (uniform DefaultMaterial*)dg.material;

    vec3f Kd = make_vec3f(0.f);
    if (!material)
        Kd = make_vec3f(dg.color);
    else
        foreach_unique(mat in material)
        {
            Kd = mat->Kd * make_vec3f(dg.color);
            if (valid(mat->map_Kd))
            {
                const vec4f diffuseColorFromMap = get4f(mat->map_Kd, dg);
                Kd = make_vec3f(dg.color * diffuseColorFromMap);
            }
        }

    // Z-Depth
    sample.z = ray.t;
    sample.alpha = dg.color.w;

    return make_vec3f(Kd);
}

void AlbedoRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData, varying ScreenSample& sample)
{
    uniform AlbedoRenderer* uniform self = (uniform AlbedoRenderer * uniform) _self;
    sample.ray.time = inf;
    sample.rgb = AlbedoRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform AlbedoRenderer_create(void* uniform cppE)
{
    uniform AlbedoRenderer* uniform self = uniform new uniform AlbedoRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = AlbedoRenderer_renderSample;
    return self;
}

export void AlbedoRenderer_set(void* uniform _self, const uniform int spp)
{
    uniform AlbedoRenderer* uniform self = (uniform AlbedoRenderer * uniform) _self;

    self->super.spp = spp;
}
