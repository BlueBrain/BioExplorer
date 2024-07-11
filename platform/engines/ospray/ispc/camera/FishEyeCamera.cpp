/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include "FishEyeCamera.h"
#include <limits>

#include "FishEyeCamera_ispc.h"

#include <platform/core/common/Properties.h>

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
FishEyeCamera::FishEyeCamera()
{
    ispcEquivalent = ::ispc::FishEyeCamera_create(this);
}

void FishEyeCamera::commit()
{
    Camera::commit();

    aspect = getParamf(CAMERA_PROPERTY_ASPECT_RATIO.name.c_str(), DEFAULT_CAMERA_ASPECT_RATIO);
    enableClippingPlanes =
        getParam(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES.name.c_str(), DEFAULT_CAMERA_ENABLE_CLIPPING_PLANES);
    clipPlanes = enableClippingPlanes ? getParamData(CAMERA_PROPERTY_CLIPPING_PLANES, nullptr) : nullptr;
    apertureRadius = getParamf(CAMERA_PROPERTY_APERTURE_RADIUS.name.c_str(), DEFAULT_CAMERA_APERTURE_RADIUS);
    focalDistance = getParamf(CAMERA_PROPERTY_FOCAL_DISTANCE.name.c_str(), DEFAULT_CAMERA_FOCAL_DISTANCE);
    useHardwareRandomizer = getParam(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER.name.c_str(),
                                     static_cast<int>(DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER));
    stereoMode = getParam(CAMERA_PROPERTY_STEREO.name.c_str(), static_cast<int>(DEFAULT_CAMERA_STEREO))
                     ? CameraStereoMode::side_by_side
                     : CameraStereoMode::mono;
    interpupillaryDistance =
        getParamf(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name.c_str(), DEFAULT_CAMERA_INTERPUPILLARY_DISTANCE);

    dir = normalize(dir);
    ::ospray::vec3f dirU = normalize(cross(dir, up));
    ::ospray::vec3f dirV = cross(dirU, dir);

    ::ospray::vec3f org = pos;
    const ::ospray::vec3f ipd_offset = 0.5f * interpupillaryDistance * dirU;

    switch (stereoMode)
    {
    case CameraStereoMode::left:
        org -= ipd_offset;
        break;
    case CameraStereoMode::right:
        org += ipd_offset;
        break;
    case CameraStereoMode::side_by_side:
        aspect *= 0.5f;
        break;
    case CameraStereoMode::mono:
        break;
    }

    if (apertureRadius > 0.f)
    {
        dirU *= focalDistance;
        dirV *= focalDistance;
        dir *= focalDistance;
    }

    const auto clipPlaneData = clipPlanes ? clipPlanes->data : nullptr;
    const size_t numClipPlanes = clipPlanes ? clipPlanes->numItems : 0;

    const auto invDir = -dir;
    ::ispc::FishEyeCamera_set(getIE(), (const ::ispc::vec3f&)org, (const ::ispc::vec3f&)invDir,
                              (const ::ispc::vec3f&)dirU, (const ::ispc::vec3f&)dirV,
                              (const ::ispc::vec4f*)clipPlaneData, numClipPlanes, apertureRadius,
                              (const ::ispc::vec3f&)ipd_offset, stereoMode, useHardwareRandomizer);
}

OSP_REGISTER_CAMERA(FishEyeCamera, fisheye);
} // namespace ospray
} // namespace engine
} // namespace core