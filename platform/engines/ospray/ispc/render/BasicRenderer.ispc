/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Based on OSPRay implementation
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

#include "BasicRenderer.ih"

#include <platform/engines/ospray/ispc/camera/utils/CameraUtils.ih>

/**
    Renderer a pixel color according to a given location in the screen space.
    @param self Pointer to current renderer
    @param sample Screen sample containing information about the ray, and the
           location in the screen space.
*/
inline vec3f BasicRenderer_shadeRay(const uniform BasicRenderer* uniform self, varying ScreenSample& sample)
{
    Ray ray = sample.ray;
    varying vec3f color = make_vec3f(0.f);

    sample.z = inf;
    sample.alpha = 0.f;

    traceRay(self->abstract.super.model, ray);

    if (ray.geomID < 0)
    {
        // No intersection, use environment map or background color
        vec4f bgSample = skyboxMapping((Renderer*)self, ray, self->abstract.bgMaterial);
        sample.alpha = bgSample.w;
        return make_vec3f(bgSample);
    }

    DifferentialGeometry dg;
    postIntersect(self->abstract.super.model, dg, ray,
                  DG_NS | DG_NG | DG_NORMALIZE | DG_FACEFORWARD | DG_MATERIALID | DG_COLOR | DG_TEXCOORD);

    uniform AdvancedMaterial* material = (uniform AdvancedMaterial*)dg.material;

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

    // Head-light shading
    color = Kd * max(0.f, dot(neg(ray.dir), dg.Ns));
    sample.alpha = 1.f;

    // Z-Depth
    sample.z = ray.t;
    sample.alpha = 1.f;

    return color;
}

void BasicRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData,
                                varying ScreenSample& sample)
{
    uniform BasicRenderer* uniform self = (uniform BasicRenderer * uniform) _self;
    sample.ray.time = self->abstract.timestamp;
    if (self->abstract.anaglyphEnabled)
    {
        ScreenSample s = sample;
        s.ray.org = s.ray.org - self->abstract.anaglyphIpdOffset;
        const vec3f colorLeft = BasicRenderer_shadeRay(self, s);
        const vec3f leftAnaglyphColor = getAnaglyphLeftColor(colorLeft);

        s = sample;
        s.ray.org = s.ray.org + self->abstract.anaglyphIpdOffset;
        const vec3f colorRight = BasicRenderer_shadeRay(self, s);
        const vec3f rightAnaglyphColor = getAnaglyphRightColor(colorRight);

        sample.rgb = leftAnaglyphColor + rightAnaglyphColor;
    }
    else
        sample.rgb = BasicRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform BasicRenderer_create(void* uniform cppE)
{
    uniform BasicRenderer* uniform self = uniform new uniform BasicRenderer;
    Renderer_Constructor(&self->abstract.super, cppE);
    self->abstract.super.renderSample = BasicRenderer_renderSample;
    return self;
}

export void BasicRenderer_set(void* uniform _self, void* uniform bgMaterial, const uniform float& timestamp,
                              const uniform int& spp, void** uniform lights, uniform int32 numLights,
                              const uniform bool anaglyphEnabled, const uniform vec3f& anaglyphIpdOffset)
{
    uniform BasicRenderer* uniform self = (uniform BasicRenderer * uniform) _self;

    self->abstract.anaglyphEnabled = anaglyphEnabled;
    self->abstract.anaglyphIpdOffset = anaglyphIpdOffset;

    self->abstract.bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;
    self->abstract.timestamp = timestamp;
    self->abstract.super.spp = spp;

    self->abstract.lights = (const uniform Light* uniform* uniform)lights;
    self->abstract.numLights = numLights;
}
