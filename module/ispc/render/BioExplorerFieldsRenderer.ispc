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

#include "BioExplorerFieldsRenderer.ih"

#include "utils/BioExplorerRandomGenerator.ih"

const float EPSILON = 1e-6f;

float treeWalker(const uniform BioExplorerFieldsRenderer* uniform self,
                 const vec3f& point, const varying float distance,
                 const varying float cutoff, const varying int32 index)
{
    const uint32 begin = self->userData[self->startIndices + index * 2];
    const uint32 end = self->userData[self->startIndices + index * 2 + 1];
    const uint32 idxData = self->startData + index * 4;

    if (idxData >= self->userDataSize)
        return 0.f;

    if (begin == 0 && end == 0)
        return self->userData[idxData + 3] / (distance * distance);

    float voxelValue = 0.f;
    for (unsigned int32 childIndex = begin; childIndex <= end; ++childIndex)
    {
        const vec3f childCenter =
            make_vec3f(self->userData[idxData], self->userData[idxData + 1],
                       self->userData[idxData + 2]);
        const vec3f delta = point - childCenter;

        const float d =
            sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);

        if (d >= cutoff)
            // Child is further than the cutoff distance, no need to evaluate
            // events in the child node, we take the precomputed value of node
            // instead
            voxelValue += self->userData[idxData + 3] / (d * d);
        else
            // Dive into the child node and compute its contents
            voxelValue += treeWalker(self, point, d, cutoff / 2.f, childIndex);
    }
    return voxelValue;
}

inline vec3f BioExplorerFieldsRenderer_shadeRay(
    const uniform BioExplorerFieldsRenderer* uniform self,
    varying ScreenSample& sample)
{
    const Ray ray = sample.ray;

    if (sample.sampleID.z == 0)
    {
        sample.alpha = 0.f;
        sample.z = inf;
    }

    float t0, t1;
    const uniform box3f box =
        make_box3f(self->offset, self->offset + self->size * self->spacing);

    intersectBox(ray, box, t0, t1);
    t0 = max(self->minRayStep, t0);

    const vec3f s = self->size * self->spacing;
    const float diag = max(max(s.x, s.y), s.z);
    const float step = max(self->minRayStep, diag / self->nbRaySteps);

    float t = t0 + (sample.sampleID.z % self->nbRayRefinementSteps) * step /
                       self->nbRayRefinementSteps;

    vec4f finalColor = make_vec4f(0.f);
    bool firstVisibleVoxel = true;

    while (finalColor.w < 0.999f && t < t1)
    {
        const vec3f p = ray.org + t * ray.dir;
        const vec3f point = (p - self->offset) / self->spacing;

        const float value =
            treeWalker(self, point, self->distance, self->cutoff, 0);

        const float sampleOpacity =
            self->transferFunction->getOpacityForValue(self->transferFunction,
                                                       value);
        if (sampleOpacity > 0.f)
        {
            const vec3f sampleColor =
                self->transferFunction->getColorForValue(self->transferFunction,
                                                         value);
            const float alpha = self->alphaCorrection * sampleOpacity;
            finalColor =
                make_vec4f(make_vec3f(finalColor) * finalColor.w +
                               (1.f - finalColor.w) * sampleColor * alpha,
                           finalColor.w + sampleOpacity);

            if (firstVisibleVoxel)
            {
                sample.alpha = 1.f;
                sample.z = min(t, sample.z);
            }
            firstVisibleVoxel = false;
        }

        t += step;
    }

    return clamp(make_vec3f(finalColor) * self->exposure, make_vec3f(0.f),
                 make_vec3f(1.f));
}

void BioExplorerFieldsRenderer_renderSample(uniform Renderer* uniform _self,
                                            void* uniform perFrameData,
                                            varying ScreenSample& sample)
{
    uniform BioExplorerFieldsRenderer* uniform self =
        (uniform BioExplorerFieldsRenderer * uniform) _self;
    sample.ray.time = self->timestamp;
    sample.rgb = BioExplorerFieldsRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform BioExplorerFieldsRenderer_create(void* uniform cppE)
{
    uniform BioExplorerFieldsRenderer* uniform self =
        uniform new uniform BioExplorerFieldsRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = BioExplorerFieldsRenderer_renderSample;
    return self;
}

export void BioExplorerFieldsRenderer_set(
    void* uniform _self, void* uniform bgMaterial,
    uniform float* uniform userData, const uniform uint64 userDataSize,
    const uniform int& randomNumber, const uniform float& timestamp,
    const uniform int& spp, void** uniform lights,
    const uniform int32 numLights, const uniform float& minRayStep,
    const uniform int32& nbRaySteps, const uniform int32& nbRayRefinementSteps,
    const uniform float& exposure, const uniform bool& useHardwareRandomizer,
    const uniform float& cutoff, const uniform float& alphaCorrection)
{
    uniform BioExplorerFieldsRenderer* uniform self =
        (uniform BioExplorerFieldsRenderer * uniform) _self;

    // OSPRay
    self->super.spp = spp;

    self->lights = (const uniform Light* uniform* uniform)lights;
    self->numLights = numLights;

    // Bio explorer
    self->bgMaterial = (uniform BioExplorerMaterial * uniform) bgMaterial;

    self->timestamp = timestamp;
    self->exposure = exposure;
    self->useHardwareRandomizer = useHardwareRandomizer;

    self->minRayStep = minRayStep;
    self->nbRaySteps = nbRaySteps;
    self->nbRayRefinementSteps = nbRayRefinementSteps;
    self->alphaCorrection = alphaCorrection;

    self->cutoff = cutoff;

    self->userData = userData;
    self->userDataSize = userDataSize;

    self->offset = (userData ? make_vec3f(userData[0], userData[1], userData[2])
                             : make_vec3f(0.f));
    self->spacing =
        (userData ? make_vec3f(userData[3], userData[4], userData[5])
                  : make_vec3f(0.f));
    self->size = (userData ? make_vec3f(userData[6], userData[7], userData[8])
                           : make_vec3f(0.f));
    self->distance = (userData ? userData[9] * 5.f : 0.f); // Octree size * 5
    self->startIndices = 11;
    self->startData = (userData ? self->startIndices + userData[10] : 0);
}

export void BioExplorerFieldsRenderer_setTransferFunction(void* uniform _self,
                                                          void* uniform value)
{
    uniform BioExplorerFieldsRenderer* uniform self =
        (uniform BioExplorerFieldsRenderer * uniform) _self;
    self->transferFunction = (TransferFunction * uniform) value;
}
