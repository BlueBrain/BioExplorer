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

#include "VectorFieldsRenderer.ih"

#include <platform/engines/ospray/ispc/camera/utils/CameraUtils.ih>
#include <platform/engines/ospray/ispc/render/utils/RendererUtils.ih>

#define DATA_SIZE 6

vec3f treeWalker(const uniform VectorFieldsRenderer* uniform self, const vec3f& point, const varying float distance,
                 const varying float cutoff, const varying int32 index = 0)
{
    const uint64 begin = self->userData[self->startIndices + index * 2];
    const uint64 end = self->userData[self->startIndices + index * 2 + 1];
    const uint64 idxData = self->startData + index * DATA_SIZE;

    if (idxData >= self->userDataSize)
        return make_vec3f(0.f);

    if (begin == 0 && end == 0)
    {
        // Leaf
        const vec3f vectorDirection =
            make_vec3f(self->userData[idxData + 3], self->userData[idxData + 4], self->userData[idxData + 5]);
        return vectorDirection / (distance * distance);
    }

    vec3f voxelValue = make_vec3f(0.f);
    for (unsigned int64 childIndex = begin; childIndex <= end; ++childIndex)
    {
        const uint64 idx = self->startData + childIndex * DATA_SIZE;
        const vec3f childPosition = make_vec3f(self->userData[idx], self->userData[idx + 1], self->userData[idx + 2]);
        const float d = length(point - childPosition);

        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate events in the child node, we take the
            // precomputed value of node  instead
            const vec3f vectorDirection =
                make_vec3f(self->userData[idx + 3], self->userData[idx + 4], self->userData[idx + 5]);
            voxelValue = voxelValue + vectorDirection / (d * d);
        }
        else
            // Dive into the child node and compute its contents
            voxelValue = voxelValue + treeWalker(self, point, d, cutoff / 2.f, childIndex);
    }
    return voxelValue;
}

inline vec3f VectorFieldsRenderer_shadeRay(const uniform VectorFieldsRenderer* uniform self,
                                           varying ScreenSample& sample)
{
    const Ray ray = sample.ray;

    if (sample.sampleID.z == 0)
    {
        sample.alpha = 0.f;
        sample.z = inf;
    }

    const uniform box3f box = make_box3f(self->offset, self->offset + self->size * self->spacing);

    float t0, t1;
    intersectBox(ray, box, t0, t1);
    t0 = max(self->minRayStep, t0);

    const vec3f s = self->size * self->spacing;
    const float diag = min(min(s.x, s.y), s.z);
    const float step = max(self->minRayStep, diag / self->nbRaySteps);
    const float sampleStep = sample.sampleID.z % self->nbRayRefinementSteps;
    float t = t0 + sampleStep * step / self->nbRayRefinementSteps;

    vec4f finalColor = make_vec4f(0.f);
    while (finalColor.w < 0.95f && t < t1)
    {
        const vec3f p = ray.org + t * ray.dir;
        const vec3f point = (p - self->offset) / self->spacing;
        const vec3f sampleValue = treeWalker(self, point, self->distance, self->cutoff);
        const float vectorLength =
            clamp(length(sampleValue), self->transferFunction->valueRange.x, self->transferFunction->valueRange.y);
        const float mapOpacity = self->transferFunction->getOpacityForValue(self->transferFunction, vectorLength);
        if (mapOpacity > 0.f)
        {
            const vec3f mapColor = self->transferFunction->getColorForValue(self->transferFunction, vectorLength);
            if (self->showVectorDirections)
            {
                const vec3f v = normalize(sampleValue);
                const vec3f vectorColor = make_vec3f(0.5f + v.x * 0.5f, 0.5f + v.y * 0.5f, 0.5f + v.z * 0.5f);
                compose(make_vec4f(vectorColor, mapOpacity), finalColor, self->alphaCorrection);
            }
            else
                compose(make_vec4f(mapColor, mapOpacity), finalColor, self->alphaCorrection);
        }

        t += step;
    }

    sample.alpha = finalColor.w;
    return clamp(make_vec3f(finalColor) * self->exposure, make_vec3f(0.f), make_vec3f(1.f));
}

void VectorFieldsRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData,
                                       varying ScreenSample& sample)
{
    uniform VectorFieldsRenderer* uniform self = (uniform VectorFieldsRenderer * uniform) _self;
    sample.ray.time = self->super.timestamp;
    if (self->super.anaglyphEnabled)
    {
        ScreenSample s = sample;
        s.ray.org = s.ray.org - self->super.anaglyphIpdOffset;
        const vec3f colorLeft = VectorFieldsRenderer_shadeRay(self, s);
        const vec3f leftAnaglyphColor = getAnaglyphLeftColor(colorLeft);

        s = sample;
        s.ray.org = s.ray.org + self->super.anaglyphIpdOffset;
        const vec3f colorRight = VectorFieldsRenderer_shadeRay(self, s);
        const vec3f rightAnaglyphColor = getAnaglyphRightColor(colorRight);

        sample.rgb = leftAnaglyphColor + rightAnaglyphColor;
    }
    else
        sample.rgb = VectorFieldsRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform VectorFieldsRenderer_create(void* uniform cppE)
{
    uniform VectorFieldsRenderer* uniform self = uniform new uniform VectorFieldsRenderer;
    Renderer_Constructor(&self->super.super, cppE);
    self->super.super.renderSample = VectorFieldsRenderer_renderSample;
    return self;
}

export void VectorFieldsRenderer_set(void* uniform _self, void* uniform bgMaterial, uniform float* uniform userData,
                                     const uniform uint64 userDataSize, const uniform int& randomNumber,
                                     const uniform float& timestamp, const uniform int& spp, void** uniform lights,
                                     const uniform int32 numLights, const uniform float& minRayStep,
                                     const uniform int32& nbRaySteps, const uniform int32& nbRayRefinementSteps,
                                     const uniform float& exposure, const uniform bool& useHardwareRandomizer,
                                     const uniform float& cutoff, const uniform float& alphaCorrection,
                                     const uniform bool& showVectorDirections, const uniform bool& anaglyphEnabled,
                                     const uniform vec3f& anaglyphIpdOffset)
{
    uniform VectorFieldsRenderer* uniform self = (uniform VectorFieldsRenderer * uniform) _self;

    // OSPRay
    self->super.super.spp = spp;

    // Abstract
    self->super.anaglyphEnabled = anaglyphEnabled;
    self->super.anaglyphIpdOffset = anaglyphIpdOffset;
    self->super.lights = (const uniform Light* uniform* uniform)lights;
    self->super.numLights = numLights;
    self->super.bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;
    self->super.timestamp = timestamp;

    // Fields
    self->exposure = exposure;
    self->useHardwareRandomizer = useHardwareRandomizer;

    self->minRayStep = minRayStep;
    self->nbRaySteps = nbRaySteps;
    self->nbRayRefinementSteps = nbRayRefinementSteps;
    self->alphaCorrection = alphaCorrection;

    self->cutoff = cutoff;
    self->showVectorDirections = showVectorDirections;

    self->userData = userData;
    self->userDataSize = userDataSize;

    self->offset = (userData ? make_vec3f(userData[0], userData[1], userData[2]) : make_vec3f(0.f));
    self->spacing = (userData ? make_vec3f(userData[3], userData[4], userData[5]) : make_vec3f(0.f));
    self->size = (userData ? make_vec3f(userData[6], userData[7], userData[8]) : make_vec3f(0.f));
    self->distance = (userData ? userData[9] * 5.f : 0.f); // Octree size * 5
    self->startIndices = 11;
    self->startData = (userData ? self->startIndices + userData[10] : 0);
}

export void VectorFieldsRenderer_setTransferFunction(void* uniform _self, void* uniform value)
{
    uniform VectorFieldsRenderer* uniform self = (uniform VectorFieldsRenderer * uniform) _self;
    self->transferFunction = (TransferFunction * uniform) value;
}
