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

#include "PointFieldsRenderer.ih"

#include <science/common/CommonTypes.h>

#include <platform/engines/ospray/ispc/camera/utils/CameraUtils.ih>
#include <platform/engines/ospray/ispc/render/utils/RandomGenerator.ih>
#include <platform/engines/ospray/ispc/render/utils/RendererUtils.ih>

float treeWalker(const uniform PointFieldsRenderer* uniform self, const vec3f& point, const varying float distance,
                 const varying float cutoff, const varying int32 index = 0)
{
    const uint64 begin = self->userData[self->startIndices + index * 2];
    const uint64 end = self->userData[self->startIndices + index * 2 + 1];
    const uint64 idxData = self->startData + index * FIELD_POINT_DATA_SIZE;

    if (idxData >= self->userDataSize)
        return 0.f;

    if (begin == 0 && end == 0)
        // Leaf
        return self->userData[idxData + FIELD_POINT_OFFSET_VALUE] / (distance * distance);

    float voxelValue = 0.f;
    for (uint64 childIndex = begin; childIndex <= end; ++childIndex)
    {
        const uint64 idx = self->startData + childIndex * FIELD_POINT_DATA_SIZE;
        const vec3f childPosition = make_vec3f(self->userData[idx + FIELD_POINT_OFFSET_POSITION_X],
                                               self->userData[idx + FIELD_POINT_OFFSET_POSITION_Y],
                                               self->userData[idx + FIELD_POINT_OFFSET_POSITION_Z]);
        const float d = length(point - childPosition);
        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate events in the child node, we take the
            // precomputed value of node instead
            voxelValue += self->userData[idx + FIELD_POINT_OFFSET_VALUE] / (d * d);
        }
        else
            // Dive into the child node and compute its contents
            voxelValue += treeWalker(self, point, d, cutoff / 2.f, childIndex);
    }
    return voxelValue;
}

inline vec3f PointFieldsRenderer_shadeRay(const uniform PointFieldsRenderer* uniform self, varying ScreenSample& sample)
{
    vec4f finalColor = make_vec4f(0.f);
    const Ray ray = sample.ray;

    if (sample.sampleID.z == 0)
    {
        sample.alpha = 0.f;
        sample.z = inf;
    }

    const uniform box3f box = make_box3f(self->offset, self->offset + self->size * self->spacing);

    const float diag = max(max(self->size.x, self->size.y), self->size.z);
    const float t_step = max(self->minRayStep, diag / (float)self->nbRaySteps);

    float t0, t1;
    intersectBox(ray, box, t0, t1);

    const float random =
        t_step * (float)(sample.sampleID.z % self->nbRayRefinementSteps) / (float)(self->nbRayRefinementSteps);

    float t = max(0.f, t0) + random;
    while (t < t1 && finalColor.w < 1.f)
    {
        const vec3f p = ray.org + t * ray.dir;
        const vec3f point = (p - self->offset) / self->spacing;
        float value = treeWalker(self, point, self->distance, self->cutoff);
        const float sampleOpacity = self->transferFunction->getOpacityForValue(self->transferFunction, value);
        if (sampleOpacity > 0.f)
        {
            const vec3f sampleColor = self->transferFunction->getColorForValue(self->transferFunction, value);
            compose(make_vec4f(sampleColor, sampleOpacity), finalColor, self->alphaCorrection);
        }

        t += t_step;
    }

    sample.alpha = finalColor.w;
    return clamp(make_vec3f(finalColor) * self->exposure, make_vec3f(0.f), make_vec3f(1.f));
}

void PointFieldsRenderer_renderSample(uniform Renderer* uniform _self, void* uniform perFrameData,
                                      varying ScreenSample& sample)
{
    uniform PointFieldsRenderer* uniform self = (uniform PointFieldsRenderer * uniform) _self;
    sample.ray.time = self->super.timestamp;
    if (self->super.anaglyphEnabled)
    {
        ScreenSample s = sample;
        s.ray.org = s.ray.org - self->super.anaglyphIpdOffset;
        const vec3f colorLeft = PointFieldsRenderer_shadeRay(self, s);
        const vec3f leftAnaglyphColor = getAnaglyphLeftColor(colorLeft);

        s = sample;
        s.ray.org = s.ray.org + self->super.anaglyphIpdOffset;
        const vec3f colorRight = PointFieldsRenderer_shadeRay(self, s);
        const vec3f rightAnaglyphColor = getAnaglyphRightColor(colorRight);

        sample.rgb = leftAnaglyphColor + rightAnaglyphColor;
    }
    else
        sample.rgb = PointFieldsRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform PointFieldsRenderer_create(void* uniform cppE)
{
    uniform PointFieldsRenderer* uniform self = uniform new uniform PointFieldsRenderer;
    Renderer_Constructor(&self->super.super, cppE);
    self->super.super.renderSample = PointFieldsRenderer_renderSample;
    return self;
}

export void PointFieldsRenderer_set(void* uniform _self, void* uniform bgMaterial, uniform float* uniform userData,
                                    const uniform uint64 userDataSize, const uniform int& randomNumber,
                                    const uniform float& timestamp, const uniform int& spp, void** uniform lights,
                                    const uniform int32 numLights, const uniform float& minRayStep,
                                    const uniform int32& nbRaySteps, const uniform int32& nbRayRefinementSteps,
                                    const uniform float& exposure, const uniform bool& useHardwareRandomizer,
                                    const uniform float& cutoff, const uniform float& alphaCorrection,
                                    const uniform bool anaglyphEnabled, const uniform vec3f& anaglyphIpdOffset)
{
    uniform PointFieldsRenderer* uniform self = (uniform PointFieldsRenderer * uniform) _self;

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

    self->userData = userData;
    self->userDataSize = userDataSize;

    self->offset = (userData ? make_vec3f(userData[OCTREE_DATA_OFFSET_X], userData[OCTREE_DATA_OFFSET_Y],
                                          userData[OCTREE_DATA_OFFSET_Z])
                             : make_vec3f(0.f));
    self->spacing = (userData ? make_vec3f(userData[OCTREE_DATA_SPACING_X], userData[OCTREE_DATA_SPACING_Y],
                                           userData[OCTREE_DATA_SPACING_Z])
                              : make_vec3f(0.f));
    self->size = (userData ? make_vec3f(userData[OCTREE_DATA_DIMENSION_X], userData[OCTREE_DATA_DIMENSION_Y],
                                        userData[OCTREE_DATA_DIMENSION_Z])
                           : make_vec3f(0.f));
    self->distance = (userData ? userData[OCTREE_DATA_INITIAL_DISTANCE] * 5.f : 0.f); // Octree size * 5
    self->startIndices = OCTREE_DATA_INDICES;
    self->startData = (userData ? self->startIndices + userData[OCTREE_DATA_VALUES] : 0);
}

export void PointFieldsRenderer_setTransferFunction(void* uniform _self, void* uniform value)
{
    uniform PointFieldsRenderer* uniform self = (uniform PointFieldsRenderer * uniform) _self;
    self->transferFunction = (TransferFunction * uniform) value;
}
