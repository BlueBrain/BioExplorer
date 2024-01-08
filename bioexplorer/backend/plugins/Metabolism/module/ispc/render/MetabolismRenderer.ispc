/*
 * Copyright 2020-2024 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "Cloud.ih"
#include "MetabolismRenderer.ih"

#include <plugin/common/CommonTypes.h>

#include <ospray/SDK/math/sampling.ih>

#define OPACITY_EPSILON 0.001f

uint32 getLocation(const uniform MetabolismRenderer* uniform self, varying Ray& ray, float& epsilon)
{
    // Back ray
    int backLocation = extra_cellular_space;
    Ray backRay = ray;
    backRay.dir = -1.f * backRay.dir;
    traceRay(self->super.model, backRay);
    if (backRay.geomID >= 0)
    {
        DifferentialGeometry dg;
        postIntersect(self->super.model, dg, backRay, DG_NS | DG_NORMALIZE);
        uniform AdvancedMaterial* objMaterial = (uniform AdvancedMaterial*)dg.material;
        foreach_unique(mat in objMaterial) backLocation = mat->userParameter;
    }
    return backLocation;
}

inline vec3f MetabolismRenderer_shadeRay(const uniform MetabolismRenderer* uniform self, varying ScreenSample& sample)
{
    vec3f finalColor = make_vec3f(0.f);
    if (sample.sampleID.z == 0)
    {
        sample.alpha = 0.f;
        sample.z = inf;
    }

    if (self->userDataSize == 0)
        return finalColor;

    float t = self->nearPlane + ((sample.sampleID.z % self->refinementSteps) * (self->rayStep / self->refinementSteps));

    // Following intersections
    float finalAlpha = 0.f;
    float voxelAlpha = 0.f;
    vec3f voxelColor = make_vec3f(0.f);
    while (t < self->farPlane && finalAlpha < 1.f)
    {
        Ray r = sample.ray;
        r.t0 = 0.f;
        r.t = self->farPlane - t;
        r.org = sample.ray.org + t * sample.ray.dir;
        r.primID = -1;
        r.geomID = -1;
        r.instID = -1;

        float geometryEpsilon;
        const uint32 location = getLocation(self, r, geometryEpsilon);

        t += self->rayStep;

        const float concentration = self->userData[1 + location];
        if (concentration >= 1e38)
            continue;
        voxelAlpha = self->transferFunction->getOpacityForValue(self->transferFunction, concentration);
        if (voxelAlpha > OPACITY_EPSILON)
            voxelColor = self->transferFunction->getColorForValue(self->transferFunction, concentration);

        float noiseAlpha = 1.f;
        if (self->noiseFrequency > 0.f && finalAlpha > OPACITY_EPSILON)
        {
            const vec3f p = sample.ray.org + t * sample.ray.dir;
            noiseAlpha = 1.f + cloud(p * self->noiseFrequency,
                                     cos((location + self->timestamp) * 0.05f) +
                                         sin((location + self->timestamp) * 0.032f) * self->noiseAmplitude);
        }

        const float alpha = clamp(noiseAlpha * self->alphaCorrection * voxelAlpha, 0.f, 1.f);
        finalColor = finalColor + voxelColor * alpha * (1.f - finalAlpha);
        finalAlpha += alpha;
    }

    sample.alpha = finalAlpha;
    return clamp(finalColor * self->exposure, make_vec3f(0.f), make_vec3f(1.f));
}

void MetabolismRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData,
                                     varying ScreenSample& sample)
{
    uniform MetabolismRenderer* uniform self = (uniform MetabolismRenderer * uniform) _self;
    sample.ray.time = self->timestamp;
    sample.rgb = MetabolismRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform MetabolismRenderer_create(void* uniform cppE)
{
    uniform MetabolismRenderer* uniform self = uniform new uniform MetabolismRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = MetabolismRenderer_renderSample;
    return self;
}

export void MetabolismRenderer_set(void* uniform _self, void* uniform bgMaterial, const uniform int& spp,
                                   void** uniform lights, const uniform uint32 numLights,
                                   uniform float* uniform userData, const uniform uint64 userDataSize,
                                   const uniform float& nearPlane, const uniform float& farPlane,
                                   const uniform float& rayStep, const uniform uint32& refinementSteps,
                                   const uniform float& exposure, const uniform float& alphaCorrection,
                                   const uniform bool& colorMapPerRegion, const uniform float& noiseFrequency,
                                   const uniform float& noiseAmplitude)
{
    uniform MetabolismRenderer* uniform self = (uniform MetabolismRenderer * uniform) _self;

    self->super.spp = spp;

    self->lights = (const uniform Light* uniform* uniform)lights;
    self->numLights = numLights;

    self->bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;

    self->exposure = exposure;

    self->rayStep = rayStep;
    self->nearPlane = nearPlane;
    self->farPlane = farPlane;
    self->alphaCorrection = alphaCorrection;

    self->colorMapPerRegion = colorMapPerRegion;

    self->refinementSteps = refinementSteps;
    self->noiseFrequency = noiseFrequency;
    self->noiseAmplitude = noiseAmplitude;

    self->userData = userData;
    self->userDataSize = userDataSize;
    self->timestamp = 0;

    if (userDataSize > 0)
        self->timestamp = self->userData[0];
}

export void MetabolismRenderer_setTransferFunction(void* uniform _self, void* uniform value)
{
    uniform MetabolismRenderer* uniform self = (uniform MetabolismRenderer * uniform) _self;
    self->transferFunction = (TransferFunction * uniform) value;
}
