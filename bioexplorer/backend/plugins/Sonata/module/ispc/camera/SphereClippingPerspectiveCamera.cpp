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

#include "SphereClippingPerspectiveCamera.h"
#include <limits>
// ispc-side stuff
#include "SphereClippingPerspectiveCamera_ispc.h"

#include <platform/core/common/Properties.h>

#include <ospray/SDK/common/Data.h>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <math.h> // M_PI
#endif

using namespace core;

namespace ospray
{
SphereClippingPerspectiveCamera::SphereClippingPerspectiveCamera()
{
    ispcEquivalent = ispc::SphereClippingPerspectiveCamera_create(this);
}

void SphereClippingPerspectiveCamera::commit()
{
    Camera::commit();

    // ------------------------------------------------------------------
    // first, "parse" the additional expected parameters
    // ------------------------------------------------------------------
    fovy = getParamf(CAMERA_PROPERTY_FIELD_OF_VIEW.name.c_str(), DEFAULT_CAMERA_FIELD_OF_VIEW);
    aspect = getParamf(CAMERA_PROPERTY_ASPECT_RATIO.name.c_str(), DEFAULT_CAMERA_ASPECT_RATIO);
    apertureRadius = getParamf(CAMERA_PROPERTY_APERTURE_RADIUS.name.c_str(), DEFAULT_CAMERA_APERTURE_RADIUS);
    focalDistance = getParamf(CAMERA_PROPERTY_FOCAL_DISTANCE.name.c_str(), DEFAULT_CAMERA_FOCAL_DISTANCE);
    stereo = getParam(CAMERA_PROPERTY_STEREO.name.c_str(), static_cast<int>(DEFAULT_CAMERA_STEREO));
    interpupillaryDistance =
        getParamf(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name.c_str(), DEFAULT_CAMERA_INTERPUPILLARY_DISTANCE);
    enableClippingPlanes =
        getParam(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES.name.c_str(), DEFAULT_CAMERA_ENABLE_CLIPPING_PLANES);
    clipPlanes = enableClippingPlanes ? getParamData(CAMERA_PROPERTY_CLIPPING_PLANES, nullptr) : nullptr;
    useHardwareRandomizer = getParam(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER.name.c_str(),
                                     static_cast<int>(DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER));

    // ------------------------------------------------------------------
    // now, update the local precomputed values
    // ------------------------------------------------------------------
    dir = normalize(dir);
    vec3f dir_du = normalize(cross(dir, up));
    vec3f dir_dv = cross(dir_du, dir); // rotate film to be perpendicular to 'dir'

    vec3f org = pos;
    const vec3f ipd_offset = 0.5f * interpupillaryDistance * dir_du;

    if (stereo)
    {
        auto bufferTarget = getParamString(CAMERA_PROPERTY_BUFFER_TARGET);
        if (bufferTarget.length() == 2)
        {
            if (bufferTarget.at(1) == 'L')
                org -= ipd_offset;
            if (bufferTarget.at(1) == 'R')
                org += ipd_offset;
        }
    }

    float imgPlane_size_y = 2.f * tanf(deg2rad(0.5f * fovy));
    float imgPlane_size_x = imgPlane_size_y * aspect;

    dir_du *= imgPlane_size_x;
    dir_dv *= imgPlane_size_y;

    vec3f dir_00 = dir - 0.5f * dir_du - 0.5f * dir_dv;

    float scaledAperture = 0.f;
    // prescale to focal plane
    if (apertureRadius > 0.f)
    {
        dir_du *= focalDistance;
        dir_dv *= focalDistance;
        dir_00 *= focalDistance;
        scaledAperture = apertureRadius / imgPlane_size_x;
    }

    const auto clipPlaneData = clipPlanes ? clipPlanes->data : nullptr;
    const size_t numClipPlanes = clipPlanes ? clipPlanes->numItems : 0;

    ispc::SphereClippingPerspectiveCamera_set(getIE(), (const ispc::vec3f&)org, (const ispc::vec3f&)dir_00,
                                              (const ispc::vec3f&)dir_du, (const ispc::vec3f&)dir_dv, scaledAperture,
                                              aspect, (const ispc::vec3f&)ipd_offset, (const ispc::vec4f*)clipPlaneData,
                                              numClipPlanes, useHardwareRandomizer);
}

OSP_REGISTER_CAMERA(SphereClippingPerspectiveCamera, sphere_clipping_perspective);

} // namespace ospray
