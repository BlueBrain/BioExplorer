/*
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include "math/sampling.ih"
#include "ospray/SDK/camera/Camera.ih"

#include <platform/engines/ospray/ispc/render/utils/RandomGenerator.ih>

struct SphereClippingPerspectiveCamera
{
    Camera super;

    vec3f org;            //!< position of camera, already contains shift when
                          //! STEREO_{LEFT|RIGHT}
    vec3f dir_00;         //!< direction of ray with screenSample=(0,0); scaled to
                          //! focusDistance
    vec3f dir_du;         //!< delta of ray direction between two pixels in X; scaled to
                          //! focusDistance
    vec3f dir_dv;         //!< delta of ray direction between two pixels in Y; scaled to
                          //! focusDistance
    float scaledAperture; //!< radius of aperture, divided by horizontal image
                          //! plane size
    float aspect;         //!< image plane size x / y
    vec3f ipd_offset;     //!< shift of camera position for left/right eye (only
                          //! when stereo)

    // Clip planes
    const uniform vec4f* clipPlanes;
    unsigned int numClipPlanes;

    // Hardware randomizer
    bool useHardwareRandomizer;
};

inline void sphereClipRay(const uniform vec4f* clipPlanes, const unsigned int numClipPlanes,
                          const varying vec3f& position, const varying vec3f& direction, varying float& near,
                          varying float& far)
{
    if (numClipPlanes == 0)
        return;

    near = far = 0.f;

    for (unsigned int i = 0; i < numClipPlanes; ++i)
    {
        const float r = dot(direction, direction);
        if (r == 0.f)
            continue;

        const vec3f center = {clipPlanes[i].x, clipPlanes[i].y, clipPlanes[i].z};
        const float radius = clipPlanes[i].w;

        const float rd2 = 1.f / r;
        const vec3f CO = center - position;
        const float projCO = dot(CO, direction) * rd2;
        const vec3f perp = CO - projCO * direction;
        const float l2 = dot(perp, perp);
        const float r2 = sqr(radius);
        if (l2 > r2)
            continue;

        const float a = (r2 - l2) * rd2;
        if (a < 0.f)
            continue;

        float td = sqrt(a);
        const float t1 = projCO - td;
        const float t2 = projCO + td;
        near = max(near, min(t1, t2));
        far = max(far, max(t1, t2));
    }
}

void SphereClippingPerspectiveCamera_initRay(uniform Camera* uniform _self, varying Ray& ray,
                                             const varying CameraSample& sample)
{
    uniform SphereClippingPerspectiveCamera* uniform self = (uniform SphereClippingPerspectiveCamera * uniform) _self;

    vec2f screen = sample.screen;
    vec3f org = self->org;

    screen = Camera_subRegion(_self, screen);

    vec3f dir = self->dir_00 + screen.x * self->dir_du + screen.y * self->dir_dv;

    if (self->super.doesDOF)
    {
        float random = getRandomValue(self->useHardwareRandomizer, make_vec3ui(screen.x, screen.y, 0.f));
        random = (random == 0.f ? 0.001f : random);
        const vec3f llp = uniformSampleDisk(random * self->scaledAperture, sample.lens);
        // transform local lens point to focal plane (dir_XX are prescaled in
        // this case)
        const vec3f lp = (llp.x * self->dir_du) + ((llp.y * self->aspect) * self->dir_dv);
        org = org + lp;
        dir = dir - lp;
    }

    varying float nearClip = self->super.nearClip;
    varying float farClip = inf;
    const varying vec3f direction = normalize(dir);

    sphereClipRay(self->clipPlanes, self->numClipPlanes, org, direction, nearClip, farClip);

    const float time = Camera_shutterTime(_self, sample.time);
    setRay(ray, org, direction, nearClip, farClip, time);
}

/*! create a new ispc-side version of a SphereClippingPerspectiveCamera - with
   given pointer to the c-class version - and return both class pointer and
    pointer to internal data back via the supplied reference
    parameters */
export void* uniform SphereClippingPerspectiveCamera_create(void* uniform cppE)
{
    uniform SphereClippingPerspectiveCamera* uniform self = uniform new uniform SphereClippingPerspectiveCamera;
    self->super.cppEquivalent = cppE;
    self->super.initRay = SphereClippingPerspectiveCamera_initRay;
    self->super.doesDOF = false;
    self->numClipPlanes = 0;
    return self;
}

export void SphereClippingPerspectiveCamera_set(void* uniform _self, const uniform vec3f& org,
                                                const uniform vec3f& dir_00, const uniform vec3f& dir_du,
                                                const uniform vec3f& dir_dv, const uniform float scaledAperture,
                                                const uniform float aspect, const uniform vec3f& ipd_offset,
                                                const uniform vec4f clipPlanes[],
                                                const uniform unsigned int numClipPlanes,
                                                const uniform bool useHardwareRandomizer)
{
    uniform SphereClippingPerspectiveCamera* uniform self = (uniform SphereClippingPerspectiveCamera * uniform) _self;
    self->org = org;
    self->dir_00 = dir_00;
    self->dir_du = dir_du;
    self->dir_dv = dir_dv;
    self->scaledAperture = scaledAperture;
    self->super.doesDOF = scaledAperture > 0.f;
    self->aspect = aspect;
    self->ipd_offset = ipd_offset;
    self->clipPlanes = clipPlanes;
    self->numClipPlanes = numClipPlanes;
    self->useHardwareRandomizer = useHardwareRandomizer;
}
