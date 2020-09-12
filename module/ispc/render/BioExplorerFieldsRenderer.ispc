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

const float EPSILON = 0.001f;

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

// AABB-Ray intersection ( http://prideout.net/blog/?p=64 ).
inline varying bool intersectBox(
    const uniform BioExplorerFieldsRenderer* uniform self,
    const varying Ray& ray, const vec3f& aabbMin, const vec3f& aabbMax,
    varying float& t0, varying float& t1)
{
    Ray r = ray;
    // We need to avoid division by zero in "vec3 invR = 1.0 / r.Dir;"
    if (r.dir.x == 0)
        r.dir.x = EPSILON;

    if (r.dir.y == 0)
        r.dir.y = EPSILON;

    if (r.dir.z == 0)
        r.dir.z = EPSILON;

    vec3f invR = 1.f / r.dir;
    vec3f tbot = invR * (aabbMin - r.org);
    vec3f ttop = invR * (aabbMax - r.org);
    vec3f tmin = min(ttop, tbot);
    vec3f tmax = max(ttop, tbot);
    vec2f t = make_vec2f(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
    t0 = max(t.x, t.y);
    t = make_vec2f(min(tmax.x, tmax.y), min(tmax.x, tmax.z));
    t1 = min(t.x, t.y);
    return (t0 <= t1);
}

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
    {
        // here we are garanteed to be in a leaf
        float d = distance;
        if (d < 0.5f)
            d = 0.5f;

        return self->userData[idxData + 3] / (d * d);
    }

    float voxelValue = 0.f;
    for (unsigned int32 childIndex = begin; childIndex <= end; ++childIndex)
    {
        const vec3f center =
            make_vec3f(self->userData[idxData], self->userData[idxData + 1],
                       self->userData[idxData + 2]);
        const vec3f delta = point - center;

        const float childDist =
            sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);

        if (childDist >= cutoff)
            voxelValue += self->userData[idxData + 3] / (childDist * childDist);
        else
            voxelValue +=
                treeWalker(self, point, childDist, cutoff / 2.0, childIndex);
    }
    return voxelValue;
}

inline vec4f shadeVoxel(const uniform BioExplorerFieldsRenderer* uniform self,
                        varying ScreenSample& sample, const vec4f& color,
                        const vec3f& normal)
{
    vec4f result = color;
    for (uniform int i = 0; self->lights && i < self->numLights; ++i)
    {
        const uniform Light* uniform light = self->lights[i];
        const varying vec2f s = make_vec2f(0.5f);
        DifferentialGeometry dg;
        const varying Light_SampleRes lightSample = light->sample(light, dg, s);

        vec3f dir = lightSample.dir;
        if (self->softShadows > 0.f)
            dir =
                normalize(self->softShadows *
                          getRandomVector(self->useHardwareRandomizer,
                                          self->super.fb->size.x, sample,
                                          lightSample.dir, self->randomNumber));

        const float cosNL = abs(dot(normal, dir));
        result.x *= cosNL;
        result.y *= cosNL;
        result.z *= cosNL;
    }
    return result;
}

inline vec3f BioExplorerFieldsRenderer_shadeRay(
    const uniform BioExplorerFieldsRenderer* uniform self,
    varying ScreenSample& sample)
{
    float t0, t1;
    vec3f aabbMin = self->offset;
    vec3f aabbMax =
        make_vec3f(self->offset) + make_vec3f(self->dims) * self->spacing;
    if (!intersectBox(self, sample.ray, aabbMin, aabbMax, t0, t1))
        return make_vec3f(0.f);

    float pathOpacity = 1.f;
#if 1
    float t = t0 + (float)(sample.sampleID.z % 8) * self->step / 8.f;
#else
    float t = t0 + getRandomValue(self->useHardwareRandomizer, sample,
                                  self->randomNumber) *
                       self->step;
#endif

    sample.alpha = 0.f;
    sample.z = inf;

    uint32 iterations = 0;
    const vec3f dims = self->dims;
    const uint32 dim = dims.x * dims.y * dims.z;
    vec4f finalColor = make_vec4f(0.f);
    bool firstIteration = true;

    while (iterations < self->maxSteps && finalColor.w < 1.f && t < t1)
    {
        const float r = (float)iterations / (float)self->maxSteps;
        const vec3f point =
            (sample.ray.org + t * sample.ray.dir - self->offset) /
            self->spacing;

        const float value =
            treeWalker(self, point, self->distance, self->cutoff, 0);

        const float sampleOpacity =
            self->transferFunction->getOpacityForValue(self->transferFunction,
                                                       value);
        if (sampleOpacity > 0.f)
        {
            vec3f sampleColor =
                self->transferFunction->getColorForValue(self->transferFunction,
                                                         value);
            const vec4f color = make_vec4f(sampleColor, sampleOpacity);
            composite(color, finalColor, 1.f);

            if (firstIteration)
            {
                sample.alpha = 1.f;
                sample.z = t;
            }
            firstIteration = false;
        }
        t += self->step;

        ++iterations;
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
    const uniform float& shadows, const uniform float& softShadows,
    const uniform bool& shadingEnabled, const uniform int& randomNumber,
    const uniform float& timestamp, const uniform int& spp,
    const uniform bool& softnessEnabled, void** uniform lights,
    const uniform int32 numLights, const uniform float& step,
    const uniform int32& maxSteps, const uniform float& exposure,
    const uniform bool& useHardwareRandomizer, const uniform float& cutoff)
{
    uniform BioExplorerFieldsRenderer* uniform self =
        (uniform BioExplorerFieldsRenderer * uniform) _self;

    // OSPRay
    self->super.spp = spp;

    self->lights = (const uniform Light* uniform* uniform)lights;
    self->numLights = numLights;

    // Bio explorer
    self->bgMaterial = (uniform BioExplorerMaterial * uniform) bgMaterial;

    self->shadows = shadows;
    self->softShadows = softShadows;

    self->timestamp = timestamp;
    self->exposure = exposure;
    self->useHardwareRandomizer = useHardwareRandomizer;

    self->shadingEnabled = shadingEnabled;
    self->softnessEnabled = softnessEnabled;

    self->step = step;
    self->maxSteps = maxSteps;

    self->cutoff = cutoff;

    self->userData = userData;
    self->userDataSize = userDataSize;

    self->offset = (userData ? make_vec3f(userData[0], userData[1], userData[2])
                             : make_vec3f(0.f));
    self->spacing =
        (userData ? make_vec3f(userData[3], userData[4], userData[5])
                  : make_vec3f(0.f));
    self->dims = (userData ? make_vec3f(userData[6], userData[7], userData[8])
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
