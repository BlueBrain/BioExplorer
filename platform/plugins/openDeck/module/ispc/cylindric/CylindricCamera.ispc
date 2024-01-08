/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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

#include "CylindricCamera.ih"
#include "math/LinearSpace.ih"
#include "math/math.ih"
#include "math/sampling.ih"

void CylindricCamera_initRay(uniform Camera* uniform _self, varying Ray& ray, const varying CameraSample& sample)
{
    uniform CylindricCamera* uniform self = (uniform CylindricCamera * uniform) _self;

    vec2f screen = sample.screen;
    screen = Camera_subRegion(_self, screen);

    const float alpha = screen.x * -pi;
    const float y = self->imgPlane_size_y * (screen.y - 0.5f);
    vec3f dir = normalize(make_vec3f(sinf(alpha), y, -cosf(alpha)));
    const LinearSpace3f cameraSpace = make_LinearSpace3f(self->dir_cam, self->dir_dv, self->dir_du);
    dir = normalize(cameraSpace * dir);

    const float time = Camera_shutterTime(_self, sample.time);
    vec3f org = self->org;
    setRay(ray, org, normalize(dir), self->super.nearClip, inf, time);
}

/*! create a new ::ispc-side version of a OpenDeckcamera - with given pointer to
    the c-class version - and return both class pointer and pointer to internal
    data back via the supplied reference parameters. */
export void* uniform CylindricCamera_create(void* uniform cppE)
{
    uniform CylindricCamera* uniform self = uniform new uniform CylindricCamera;
    self->super.cppEquivalent = cppE;
    self->super.initRay = CylindricCamera_initRay;
    self->super.doesDOF = false;
    return self;
}

export void CylindricCamera_set(void* uniform _self, const uniform vec3f& org, const uniform vec3f& dir_cam,
                                const uniform vec3f& dir_du, const uniform vec3f& dir_dv,
                                const uniform float imgPlane_size_y)
{
    uniform CylindricCamera* uniform self = (uniform CylindricCamera * uniform) _self;
    self->org = org;
    self->dir_cam = dir_cam;
    self->dir_du = dir_du;
    self->dir_dv = dir_dv;
    self->imgPlane_size_y = imgPlane_size_y;
}
