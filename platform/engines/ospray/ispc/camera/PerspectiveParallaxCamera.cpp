/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "PerspectiveParallaxCamera.h"
#include "PerspectiveParallaxCamera_ispc.h"

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/OSPRayProperties.h>

namespace core
{
namespace engine
{
namespace ospray
{
PerspectiveParallaxCamera::PerspectiveParallaxCamera()
{
    ispcEquivalent = ::ispc::PerspectiveParallaxCamera_create(this);
}

void PerspectiveParallaxCamera::commit()
{
    Camera::commit();

    const float fieldOfView = getParamf(CAMERA_PROPERTY_FIELD_OF_VIEW.name.c_str(), DEFAULT_CAMERA_FIELD_OF_VIEW);
    float aspectRatio = getParamf(CAMERA_PROPERTY_ASPECT_RATIO.name.c_str(), DEFAULT_CAMERA_ASPECT_RATIO);

    const float interpupillaryDistance =
        getParamf(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name.c_str(), DEFAULT_CAMERA_INTERPUPILLARY_DISTANCE);
    const float zeroParallaxPlane =
        getParamf(OSPRAY_CAMERA_PROPERTY_ZERO_PARALLAX_PLANE.name.c_str(), OSPRAY_DEFAULT_CAMERA_ZERO_PARALLAX_PLANE);

    float idpOffset = 0.0f;
    auto bufferTarget = getParamString(CAMERA_PROPERTY_BUFFER_TARGET);
    if (bufferTarget.length() == 2)
    {
        if (bufferTarget.at(1) == 'L')
            idpOffset = -interpupillaryDistance * 0.5f;
        if (bufferTarget.at(1) == 'R')
            idpOffset = +interpupillaryDistance * 0.5f;
    }

    ::ospray::vec3f org = pos;
    dir = normalize(dir);
    const ::ospray::vec3f dir_du = normalize(cross(dir, up));
    const ::ospray::vec3f dir_dv = normalize(up);
    dir = -dir;

    const float imgPlane_size_y = 2.f * zeroParallaxPlane * tanf(::ospray::deg2rad(0.5f * fieldOfView));
    const float imgPlane_size_x = imgPlane_size_y * aspectRatio;

    ::ispc::PerspectiveParallaxCamera_set(getIE(), (const ::ispc::vec3f&)org, (const ::ispc::vec3f&)dir,
                                        (const ::ispc::vec3f&)dir_du, (const ::ispc::vec3f&)dir_dv, zeroParallaxPlane,
                                        imgPlane_size_y, imgPlane_size_x, idpOffset);
}

OSP_REGISTER_CAMERA(PerspectiveParallaxCamera, perspectiveParallax);
} // namespace ospray
} // namespace engine
} // namespace core