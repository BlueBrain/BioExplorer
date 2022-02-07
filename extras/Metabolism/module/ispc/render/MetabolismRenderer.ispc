/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue Brain Project / EPFL
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

#define NB_MAX_INTERSECTIONS 20
#define OPACITY_EPSILON 0.01f

bool getNextIntersection(const uniform MetabolismRenderer* uniform self,
                         varying Ray& ray, float& distance, uint32& location,
                         bool& enteringLocation, float& epsilon)
{
    traceRay(self->super.model, ray);
    if (ray.geomID >= 0)
    {
        DifferentialGeometry dg;
        postIntersect(self->super.model, dg, ray, DG_NS | DG_NORMALIZE);

        uniform AdvancedMaterial* objMaterial =
            (uniform AdvancedMaterial*)dg.material;

        foreach_unique(mat in objMaterial) location = mat->userParameter;

        distance = ray.t;
        enteringLocation = dot(ray.dir, dg.Ns) < 0.f;
        epsilon = dg.epsilon;
        return true;
    }
    return false;
}

inline vec3f MetabolismRenderer_shadeRay(
    const uniform MetabolismRenderer* uniform self,
    varying ScreenSample& sample)
{
    vec3f finalColor = make_vec3f(0.f);
    if (sample.sampleID.z == 0)
    {
        sample.alpha = 0.f;
        sample.z = inf;
    }

    if (self->userDataSize == 0)
        return finalColor;

    bool firstVisibleVoxel = true;

    // Find intersections for current ray
    float intersectionDistances[NB_MAX_INTERSECTIONS];
    uint32 intersectionLocations[NB_MAX_INTERSECTIONS];
    float intersectionOpacities[NB_MAX_INTERSECTIONS];
    uint32 stackLocations[NB_MAX_INTERSECTIONS];
    int32 stackCounts[NB_MAX_INTERSECTIONS];

    for (uint32 i = 0; i < NB_MAX_INTERSECTIONS; ++i)
    {
        intersectionDistances[i] = 0.f;
        intersectionLocations[i] = extra_cellular_space;
        intersectionOpacities[i] = 0.f;
        stackLocations[i] = 0;
        stackCounts[i] = 0;
    }

    float t = self->nearPlane + ((sample.sampleID.z % self->refinementSteps) *
                                 (self->rayStep / self->refinementSteps));

    // Initial intersection
    float distance;
    uint32 location = extra_cellular_space;
    bool enteringLocation;
    float geometryEpsilon;

    uint32 intersectionIndex = 0;
    intersectionDistances[intersectionIndex] = t;
    intersectionLocations[intersectionIndex] = location;
    intersectionOpacities[intersectionIndex] =
        self->transferFunction->getOpacityForValue(self->transferFunction,
                                                   location);

    uint32 stackIndex = 0;
    stackCounts[stackIndex] = 1;
    stackLocations[stackIndex] = location;

    // Following intersections
    Ray r = sample.ray;
    r.t0 = 0.f;
    r.t = self->farPlane;
    while (true)
    {
        if (getNextIntersection(self, r, distance, location, enteringLocation,
                                geometryEpsilon))
        {
            const float locationOpacity =
                self->transferFunction->getOpacityForValue(
                    self->transferFunction, location);
            if (locationOpacity > OPACITY_EPSILON)
            {
                if (enteringLocation)
                {
                    // Entering a location
                    if (stackLocations[stackIndex] == location)
                    {
                        // Already inside that location
                        ++stackCounts[stackIndex];
                    }
                    else
                    {
                        // Entering new location, adding intersection
                        ++stackIndex;
                        if (stackIndex > NB_MAX_INTERSECTIONS)
                            return make_vec3f(1, 0, 0);

                        stackCounts[stackIndex] = 1;
                        stackLocations[stackIndex] = location;

                        ++intersectionIndex;
                        if (intersectionIndex > NB_MAX_INTERSECTIONS)
                            return make_vec3f(0, 0, 1);

                        intersectionDistances[intersectionIndex] = distance;
                        intersectionLocations[intersectionIndex] = location;
                        intersectionOpacities[intersectionIndex] =
                            locationOpacity;
                    }
                }
                else
                {
                    // Exiting a location
                    if (stackLocations[stackIndex] == location)
                    {
                        // Already inside that location
                        --stackCounts[stackIndex];
                        if (stackCounts[stackIndex] == 0)
                        {
                            --stackIndex;
                            --stackCounts[stackIndex];

                            ++intersectionIndex;
                            if (intersectionIndex > NB_MAX_INTERSECTIONS)
                                return make_vec3f(0, 1, 0);

                            intersectionDistances[intersectionIndex] = distance;
                            intersectionLocations[intersectionIndex] = location;
                            intersectionOpacities[intersectionIndex] =
                                intersectionOpacities[intersectionIndex - 1];
                        }
                    }
                    // else Strange...
                }
            }

            // r.org = r.org + (distance + geometryEpsilon) * r.dir;
            r.t0 = distance + geometryEpsilon;
            r.t = self->farPlane - distance;
            r.primID = -1;
            r.geomID = -1;
            r.instID = -1;
        }
        else
            break;
    }

    // How many intersections on that ray?
    if (intersectionIndex == 0)
    {
        // No intersection dectected
        if (intersectionOpacities[0] < OPACITY_EPSILON)
            // Location is invisible, returning Background color
            return finalColor;

        // Location is visible, ray-march all the way from near to far plane
        intersectionDistances[0] = self->farPlane;
    }

    // Ray marching along identified intersections
    float finalAlpha = 0.f;
    uint32 currentIntersection = 0;
    uint32 lastVisibleIntersection = 0;
    while (finalAlpha < 1.f && t < self->farPlane)
    {
        if (t > intersectionDistances[currentIntersection])
            ++currentIntersection;

        const float intersectionOpacity =
            intersectionOpacities[currentIntersection];

        if (intersectionOpacity > OPACITY_EPSILON)
            lastVisibleIntersection = currentIntersection;

        const float opacity = intersectionOpacities[lastVisibleIntersection];

        float voxelAlpha = 0.f;
        vec3f voxelColor = make_vec3f(0.f);

        if (opacity > OPACITY_EPSILON)
        {
            const uint32 location =
                intersectionLocations[lastVisibleIntersection];
            const float concentration = (location < self->userDataSize - 1)
                                            ? self->userData[1 + location]
                                            : 0.f;
            voxelAlpha = concentration * opacity;
            if (voxelAlpha > OPACITY_EPSILON)
                voxelColor = self->transferFunction->getColorForValue(
                    self->transferFunction, location);

            float noiseAlpha = 1.f;
            if (self->noiseFrequency > 0.f && opacity > OPACITY_EPSILON)
            {
                const vec3f p = sample.ray.org + t * sample.ray.dir;
                noiseAlpha =
                    1.f + cloud(p * self->noiseFrequency,
                                cos((location + self->timestamp) * 0.05f) +
                                    sin((location + self->timestamp) * 0.032f) *
                                        self->noiseAmplitude);
            }

            const float alpha =
                clamp(noiseAlpha * self->alphaCorrection * voxelAlpha, 0.f,
                      1.f);
            finalColor = finalColor + voxelColor * alpha * (1.f - finalAlpha);
            finalAlpha += alpha;
        }

        t += self->rayStep;
    }

    sample.alpha = finalAlpha;
    return clamp(finalColor * self->exposure, make_vec3f(0.f), make_vec3f(1.f));
}

void MetabolismRenderer_renderSample(uniform Renderer* uniform _self,
                                     void* uniform perFrameData,
                                     varying ScreenSample& sample)
{
    uniform MetabolismRenderer* uniform self =
        (uniform MetabolismRenderer * uniform) _self;
    sample.ray.time = self->timestamp;
    sample.rgb = MetabolismRenderer_shadeRay(self, sample);
}

// Exports (called from C++)
export void* uniform MetabolismRenderer_create(void* uniform cppE)
{
    uniform MetabolismRenderer* uniform self =
        uniform new uniform MetabolismRenderer;
    Renderer_Constructor(&self->super, cppE);
    self->super.renderSample = MetabolismRenderer_renderSample;
    return self;
}

export void MetabolismRenderer_set(
    void* uniform _self, void* uniform bgMaterial, const uniform int& spp,
    void** uniform lights, const uniform uint32 numLights,
    uniform float* uniform userData, const uniform uint64 userDataSize,
    const uniform float& nearPlane, const uniform float& farPlane,
    const uniform float& rayStep, const uniform uint32& refinementSteps,
    const uniform float& exposure, const uniform float& alphaCorrection,
    const uniform float& searchLength, const uniform bool& useRandomSearch,
    const uniform float& noiseFrequency, const uniform float& noiseAmplitude)
{
    uniform MetabolismRenderer* uniform self =
        (uniform MetabolismRenderer * uniform) _self;

    self->super.spp = spp;

    self->lights = (const uniform Light* uniform* uniform)lights;
    self->numLights = numLights;

    self->bgMaterial = (uniform AdvancedMaterial * uniform) bgMaterial;

    self->exposure = exposure;

    self->rayStep = rayStep;
    self->nearPlane = nearPlane;
    self->farPlane = farPlane;
    self->alphaCorrection = alphaCorrection;

    self->useRandomSearch = useRandomSearch;
    self->searchLength = searchLength;

    self->refinementSteps = refinementSteps;
    self->noiseFrequency = noiseFrequency;
    self->noiseAmplitude = noiseAmplitude;

    self->userData = userData;
    self->userDataSize = userDataSize;
    self->timestamp = 0;

    if (userDataSize > 0)
        self->timestamp = self->userData[0];
}

export void MetabolismRenderer_setTransferFunction(void* uniform _self,
                                                   void* uniform value)
{
    uniform MetabolismRenderer* uniform self =
        (uniform MetabolismRenderer * uniform) _self;
    self->transferFunction = (TransferFunction * uniform) value;
}
