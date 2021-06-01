/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#include "FieldsRenderer.ih"

#include "utils/BioExplorerRandomGenerator.ih"

const float EPSILON = 1e-6f;

float treeWalker(const uniform FieldsRenderer* uniform self, const vec3f& point,
                 const varying float distance, const varying float cutoff,
                 const varying int32 index = 0)
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

        float d =
            sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);

        // Minimum distance
        if (d < self->spacing.x / 2.f)
            d = self->spacing.x / 2.f;

        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate
            // events in the child node, we take the precomputed value of node
            // instead
            voxelValue += self->userData[idxData + 3] / (d * d);
        }
        else
            // Dive into the child node and compute its contents
            voxelValue += treeWalker(self, point, d, cutoff / 2.f, childIndex);
    }
    return voxelValue;
}

/**
    Composes source and destination colors according to specified alpha
   correction
    @param src Source RGBA color
    @param dst Returned destination RGBA color
    @param alphaCorrection Alpha correction used to blend colors
*/
inline void composite(const varying vec4f& src, varying vec4f& dst,
                      const varying float alphaCorrection)
{
    const float alpha =
        1.f - pow(1.f - min(src.w, 1.f - 1.f / 256.f), alphaCorrection);
    const float a = alpha * (1.f - dst.w);
    dst.x = dst.x + src.x * a;
    dst.y = dst.y + src.y * a;
    dst.z = dst.z + src.z * a;
    dst.w += (alpha * (1.f - dst.w));
}

inline vec3f FieldsRenderer_shadeRay(const uniform FieldsRenderer* uniform self,
                                     varying ScreenSample& sample)
{
    const Ray ray = sample.ray;

    if (sample.sampleID.z == 0)
    {
        sample.alpha = 0.f;
        sample.z = inf;
    }

    const uniform box3f box =
        make_box3f(self->offset, self->offset + self->size * self->spacing);

    float t0, t1;
    intersectBox(ray, box, t0, t1);
    t0 = max(self->minRayStep, t0);

    const vec3f s = self->size * self->spacing;
    const float diag = max(max(s.x, s.y), s.z);
    const float step = max(self->minRayStep, diag / self->nbRaySteps);

    float t = t0 + (sample.sampleID.z % self->nbRayRefinementSteps) * step /
                       self->nbRayRefinementSteps;

    vec4f finalColor = make_vec4f(0.f);
    bool firstVisibleVoxel = true;

    while (finalColor.w < 0.9f && t < t1)
    {
        const vec3f p = ray.org + t * ray.dir;
        const vec3f point = (p - self->offset) / self->spacing;

        float value = treeWalker(self, point, self->distance, self->cutoff);

        value = clamp(value, self->transferFunction->valueRange.x,
                      self->transferFunction->valueRange.y);

        const float sampleOpacity =
            self->transferFunction->getOpacityForValue(self->transferFunction,
                                                       value);
        if (sampleOpacity > 0.f)
        {
            const vec3f sampleColor =
                self->transferFunction->getColorForValue(self->transferFunction,
                                                         value);
            const float alpha = self->alphaCorrection * sampleOpacity;
            composite(make_vec4f(sampleColor, alpha), finalColor, 1.0f);

            if (firstVisibleVoxel)
                sample.z = min(t, sample.z);

            firstVisibleVoxel = false;
        }

        t += step;
    }

    sample.alpha = finalColor.w;
    return clamp(make_vec3f(finalColor) * self->exposure, make_vec3f(0.f),
                 make_vec3f(1.f));
}

void FieldsRenderer_renderSample(uniform Renderer* uniform _self,
                                 void* uniform perFrameData,
                                 varying ScreenSample& sample)
{
    uniform FieldsRenderer* uniform self =
        (uniform FieldsRenderer * uniform) _self;
    sample.ray.time = self->timestamp;
    sample.rgb = FieldsRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform FieldsRenderer_create(void* uniform cppE)
{
    uniform FieldsRenderer* uniform self = uniform new uniform FieldsRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = FieldsRenderer_renderSample;
    return self;
}

export void FieldsRenderer_set(
    void* uniform _self, void* uniform bgMaterial,
    uniform float* uniform userData, const uniform uint64 userDataSize,
    const uniform int& randomNumber, const uniform float& timestamp,
    const uniform int& spp, void** uniform lights,
    const uniform int32 numLights, const uniform float& minRayStep,
    const uniform int32& nbRaySteps, const uniform int32& nbRayRefinementSteps,
    const uniform float& exposure, const uniform bool& useHardwareRandomizer,
    const uniform float& cutoff, const uniform float& alphaCorrection)
{
    uniform FieldsRenderer* uniform self =
        (uniform FieldsRenderer * uniform) _self;

    // OSPRay
    self->super.spp = spp;

    self->lights = (const uniform Light* uniform* uniform)lights;
    self->numLights = numLights;

    // Bio explorer
    self->bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;

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

export void FieldsRenderer_setTransferFunction(void* uniform _self,
                                               void* uniform value)
{
    uniform FieldsRenderer* uniform self =
        (uniform FieldsRenderer * uniform) _self;
    self->transferFunction = (TransferFunction * uniform) value;
}
