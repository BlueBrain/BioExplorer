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

#include "DepthRenderer.ih"

#include <ospray/SDK/common/Material.ih>

inline vec3f DepthRenderer_shadeRay(const uniform DepthRenderer* uniform self, varying ScreenSample& sample)
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

void DepthRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData, varying ScreenSample& sample)
{
    uniform DepthRenderer* uniform self = (uniform DepthRenderer * uniform) _self;
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

export void DepthRenderer_set(void* uniform _self, const uniform int spp, const uniform float infinity)
{
    uniform DepthRenderer* uniform self = (uniform DepthRenderer * uniform) _self;

    self->super.spp = spp;
    self->infinity = infinity;
}
