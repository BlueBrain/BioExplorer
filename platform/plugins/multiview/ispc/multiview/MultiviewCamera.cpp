/*
 * Copyright (c) 2018-2023, EPFL/Blue Brain Project
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

#include "MultiviewCamera.h"
#include "MultiviewCamera_ispc.h"

#include <platform/core/common/Properties.h>
#include <platform/plugins/multiview/common/CommonStructs.h>

#include <ospray/SDK/common/Data.h>

using namespace core;

namespace ospray
{
MultiviewCamera::MultiviewCamera()
{
    ispcEquivalent = ::ispc::MultiviewCamera_create(this);
}

std::string MultiviewCamera::toString() const
{
    return "ospray::MultiviewCamera";
}

void MultiviewCamera::commit()
{
    Camera::commit();

    const float fovy = getParamf(CAMERA_PROPERTY_FIELD_OF_VIEW.name.c_str(), DEFAULT_CAMERA_FIELD_OF_VIEW);
    const float aspect = getParamf(CAMERA_PROPERTY_ASPECT_RATIO.name.c_str(), 1.66667f);
    const float apertureRadius =
        getParamf(CAMERA_PROPERTY_APERTURE_RADIUS.name.c_str(), DEFAULT_CAMERA_APERTURE_RADIUS);
    const float focalDistance = getParamf(CAMERA_PROPERTY_FOCAL_DISTANCE.name.c_str(), DEFAULT_CAMERA_FOCAL_DISTANCE);
    const float height = getParamf(CAMERA_PROPERTY_HEIGHT.name.c_str(), DEFAULT_CAMERA_HEIGHT);
    const float armLength = getParamf(PARAM_ARM_LENGTH.c_str(), DEFAULT_PARAM_ARM_LENGTH);

    clipPlanes = getParamData(CAMERA_PROPERTY_CLIPPING_PLANES, nullptr);

    dir = normalize(dir);
    vec3f dir_du = normalize(cross(dir, up));
    vec3f dir_dv = cross(dir_du, dir);

    vec3f org = pos;

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

    ::ispc::MultiviewCamera_set(getIE(), (const ::ispc::vec3f&)org, (const ::ispc::vec3f&)dir_00, (const ::ispc::vec3f&)dir_du,
                              (const ::ispc::vec3f&)dir_dv, scaledAperture, height, aspect, armLength,
                              (const ::ispc::vec4f*)clipPlaneData, numClipPlanes);
}

OSP_REGISTER_CAMERA(MultiviewCamera, multiview);
} // namespace ospray
