/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Grigori Chevtchenko <grigori.chevtchenko@epfl.ch>
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "PanoramicCamera.h"
#include <limits>

#include "PanoramicCamera_ispc.h"

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/OSPRayProperties.h>

#include <ospray/SDK/common/Data.h>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <math.h> // M_PI
#endif

namespace core
{
namespace engine
{
namespace ospray
{
PanoramicCamera::PanoramicCamera()
{
    ispcEquivalent = ::ispc::PanoramicCamera_create(this);
}

void PanoramicCamera::commit()
{
    Camera::commit();

    stereo = getParam(CAMERA_PROPERTY_STEREO.name.c_str(), static_cast<int>(DEFAULT_CAMERA_STEREO));
    half = getParam(OSPRAY_CAMERA_PROPERTY_HALF_SPHERE.name.c_str(), OSPRAY_DEFAULT_CAMERA_HALF_SPHERE);
    interpupillaryDistance =
        getParamf(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name.c_str(), DEFAULT_CAMERA_INTERPUPILLARY_DISTANCE);
    enableClippingPlanes =
        getParam(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES.name.c_str(), DEFAULT_CAMERA_ENABLE_CLIPPING_PLANES);
    clipPlanes = enableClippingPlanes ? getParamData(CAMERA_PROPERTY_CLIPPING_PLANES, nullptr) : nullptr;

    dir = normalize(dir);
    ::ospray::vec3f dirU = normalize(cross(dir, up));
    ::ospray::vec3f dirV = cross(dirU, dir); // rotate film to be perpendicular to 'dir'
    ::ospray::vec3f org = pos;
    const ::ospray::vec3f ipd_offset = 0.5f * interpupillaryDistance * dirU;

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

    const auto clipPlaneData = clipPlanes ? clipPlanes->data : nullptr;
    const size_t numClipPlanes = clipPlanes ? clipPlanes->numItems : 0;

    ::ispc::PanoramicCamera_set(getIE(), (const ::ispc::vec3f&)org, (const ::ispc::vec3f&)dir,
                                (const ::ispc::vec3f&)dirU, (const ::ispc::vec3f&)dirV,
                                (const ::ispc::vec3f&)ipd_offset, (const ::ispc::vec4f*)clipPlaneData, numClipPlanes,
                                half);
}

OSP_REGISTER_CAMERA(PanoramicCamera, panoramic);
} // namespace ospray
} // namespace engine
} // namespace core