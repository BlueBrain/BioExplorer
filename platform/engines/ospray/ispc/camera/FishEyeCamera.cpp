/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
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

#include "FishEyeCamera.h"
#include <limits>

#include "FishEyeCamera_ispc.h"

#include <platform/core/common/Properties.h>

#include <ospray/SDK/common/Data.h>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <math.h> // M_PI
#endif

using namespace core;

namespace ospray
{
FishEyeCamera::FishEyeCamera()
{
    ispcEquivalent = ispc::FishEyeCamera_create(this);
}

void FishEyeCamera::commit()
{
    Camera::commit();

    enableClippingPlanes =
        getParam(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES.name.c_str(), DEFAULT_CAMERA_ENABLE_CLIPPING_PLANES);
    clipPlanes = enableClippingPlanes ? getParamData(CAMERA_PROPERTY_CLIPPING_PLANES, nullptr) : nullptr;
    apertureRadius = getParamf(CAMERA_PROPERTY_APERTURE_RADIUS.name.c_str(), DEFAULT_CAMERA_APERTURE_RADIUS);
    focalDistance = getParamf(CAMERA_PROPERTY_FOCAL_DISTANCE.name.c_str(), DEFAULT_CAMERA_FOCAL_DISTANCE);
    useHardwareRandomizer =
        getParam(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER.name.c_str(), DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER);

    dir = normalize(dir);
    vec3f dirU = normalize(cross(dir, up));
    vec3f dirV = cross(dirU, dir);

    vec3f org = pos;

    if (apertureRadius > 0.f)
    {
        dirU *= focalDistance;
        dirV *= focalDistance;
        dir *= focalDistance;
    }

    const auto clipPlaneData = clipPlanes ? clipPlanes->data : nullptr;
    const size_t numClipPlanes = clipPlanes ? clipPlanes->numItems : 0;

    const auto invDir = -dir;
    ispc::FishEyeCamera_set(getIE(), (const ispc::vec3f&)org, (const ispc::vec3f&)invDir, (const ispc::vec3f&)dirU,
                            (const ispc::vec3f&)dirV, (const ispc::vec4f*)clipPlaneData, numClipPlanes, apertureRadius,
                            useHardwareRandomizer);
}

OSP_REGISTER_CAMERA(FishEyeCamera, fisheye);
} // namespace ospray
