/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include "utils.ih"

#include "math/sampling.ih"
#include "ospray/SDK/camera/Camera.ih"

inline float getRandomValue()
{
    float r;
    int nbMaxTries = 10;
    while (nbMaxTries >= 0 && rdrand(&r) == false)
        --nbMaxTries;
    return r;
}

struct PerspectiveStereoCamera
{
    Camera super;

    vec3f org;
    vec3f dir_00;
    vec3f dir_du;
    vec3f dir_dv;
    float scaledAperture;
    float aspect;
    bool side_by_side;
    vec3f ipd_offset;

    // Clip planes
    const uniform vec4f* clipPlanes;
    unsigned int numClipPlanes;
};

void PerspectiveStereoCamera_initRay(uniform Camera* uniform _self,
                                     varying Ray& ray,
                                     const varying CameraSample& sample)
{
    uniform PerspectiveStereoCamera* uniform self =
        (uniform PerspectiveStereoCamera * uniform) _self;

    vec2f screen = sample.screen;
    vec3f org = self->org;

    if (self->side_by_side)
    {
        screen.x *= 2.f;
        if (screen.x < 1.f)
        {
            org = org - self->ipd_offset;
        }
        else
        {
            org = org + self->ipd_offset;
            screen.x -= 1.f;
        }
    }

    screen = Camera_subRegion(_self, screen);

    vec3f dir =
        self->dir_00 + screen.x * self->dir_du + screen.y * self->dir_dv;

    if (self->super.doesDOF)
    {
        float random = 1.f;
        random = getRandomValue();
        random = (random == 0.f ? 0.001f : random);
        const vec3f llp =
            uniformSampleDisk(random * self->scaledAperture, sample.lens);
        // transform local lens point to focal plane (dir_XX are prescaled in
        // this case)
        const vec3f lp =
            (llp.x * self->dir_du) + ((llp.y * self->aspect) * self->dir_dv);
        org = org + lp;
        dir = dir - lp;
    }

    float nearClip = self->super.nearClip;
    float farClip = inf;
    const varying vec3f direction = normalize(dir);

    clipRay(self->clipPlanes, self->numClipPlanes, org, direction, nearClip,
            farClip);

    const float time = Camera_shutterTime(_self, sample.time);
    setRay(ray, org, direction, nearClip, farClip, time);
}

export void* uniform PerspectiveStereoCamera_create(void* uniform cppE)
{
    uniform PerspectiveStereoCamera* uniform self =
        uniform new uniform PerspectiveStereoCamera;
    self->super.cppEquivalent = cppE;
    self->super.initRay = PerspectiveStereoCamera_initRay;
    self->super.doesDOF = false;
    self->numClipPlanes = 0;
    return self;
}

export void PerspectiveStereoCamera_set(
    void* uniform _self, const uniform vec3f& org, const uniform vec3f& dir_00,
    const uniform vec3f& dir_du, const uniform vec3f& dir_dv,
    const uniform float scaledAperture, const uniform float aspect,
    const uniform vec3f& ipd_offset, const uniform bool side_by_side,
    const uniform vec4f clipPlanes[], const uniform unsigned int numClipPlanes,
    const uniform float nearClip)
{
    uniform PerspectiveStereoCamera* uniform self =
        (uniform PerspectiveStereoCamera * uniform) _self;
    self->org = org;
    self->dir_00 = dir_00;
    self->dir_du = dir_du;
    self->dir_dv = dir_dv;
    self->scaledAperture = scaledAperture;
    self->super.doesDOF = scaledAperture > 0.f;
    self->aspect = aspect;
    self->side_by_side = side_by_side;
    self->ipd_offset = ipd_offset;
    self->clipPlanes = clipPlanes;
    self->numClipPlanes = numClipPlanes;
    self->super.nearClip = nearClip;
}
