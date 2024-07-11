/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "CylindricStereoCamera.h"
#include "CylindricStereoCamera_ispc.h"

#include <platform/core/common/Properties.h>

using namespace core;

namespace
{
constexpr float OPENDECK_FOV_Y = 48.549f;
} // namespace

namespace ospray
{
CylindricStereoCamera::CylindricStereoCamera()
{
    ispcEquivalent = ::ispc::CylindricStereoCamera_create(this);
}

std::string CylindricStereoCamera::toString() const
{
    return "ospray::CylindricStereoCamera";
}

void CylindricStereoCamera::commit()
{
    Camera::commit();

    const auto stereoMode = getStereoMode();
    const auto ipd = getInterpupillaryDistance(stereoMode);
    const auto sideBySide = stereoMode == StereoMode::OSP_STEREO_SIDE_BY_SIDE;

    dir = normalize(dir);
    // The tracking model of the 3d glasses is inversed
    // so we need to negate dir_du here.
    const auto dir_du = -normalize(cross(dir, up));
    const auto dir_dv = normalize(up);
    dir = -dir;

    const auto imgPlane_size_y = 2.0f * tanf(deg2rad(0.5f * OPENDECK_FOV_Y));

    ::ispc::CylindricStereoCamera_set(getIE(), (const ::ispc::vec3f&)pos, (const ::ispc::vec3f&)dir,
                                      (const ::ispc::vec3f&)dir_du, (const ::ispc::vec3f&)dir_dv, imgPlane_size_y, ipd,
                                      sideBySide);
}

CylindricStereoCamera::StereoMode CylindricStereoCamera::getStereoMode()
{
    return static_cast<StereoMode>(getParam1i("stereoMode", StereoMode::OSP_STEREO_SIDE_BY_SIDE));
}

float CylindricStereoCamera::getInterpupillaryDistance(const StereoMode stereoMode)
{
    const auto interpupillaryDistance =
        getParamf(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name.c_str(), DEFAULT_CAMERA_INTERPUPILLARY_DISTANCE);

    switch (stereoMode)
    {
    case StereoMode::OSP_STEREO_LEFT:
        return -interpupillaryDistance;
    case StereoMode::OSP_STEREO_RIGHT:
        return interpupillaryDistance;
    case StereoMode::OSP_STEREO_SIDE_BY_SIDE:
        return interpupillaryDistance;
    case StereoMode::OSP_STEREO_NONE:
    default:
        return 0.0f;
    }
}

OSP_REGISTER_CAMERA(CylindricStereoCamera, cylindricStereo);
} // namespace ospray
