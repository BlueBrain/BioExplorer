/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "AnaglyphCamera.h"
#include <limits>

#include "AnaglyphCamera_ispc.h"

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
AnaglyphCamera::AnaglyphCamera()
{
    ispcEquivalent = ::ispc::AnaglyphCamera_create(this);
}

void AnaglyphCamera::commit()
{
    Camera::commit();

    fieldOfView = getParamf(CAMERA_PROPERTY_FIELD_OF_VIEW.name.c_str(), DEFAULT_CAMERA_FIELD_OF_VIEW);
    aspect = getParamf(CAMERA_PROPERTY_ASPECT_RATIO.name.c_str(), DEFAULT_CAMERA_ASPECT_RATIO);
    apertureRadius = getParamf(CAMERA_PROPERTY_APERTURE_RADIUS.name.c_str(), DEFAULT_CAMERA_APERTURE_RADIUS);
    focalDistance = getParamf(CAMERA_PROPERTY_FOCAL_DISTANCE.name.c_str(), DEFAULT_CAMERA_FOCAL_DISTANCE);
    nearClip = getParamf(CAMERA_PROPERTY_NEAR_CLIP.name.c_str(), DEFAULT_CAMERA_NEAR_CLIP);
    interpupillaryDistance =
        getParamf(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name.c_str(), DEFAULT_CAMERA_INTERPUPILLARY_DISTANCE);
    enableClippingPlanes =
        getParam(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES.name.c_str(), DEFAULT_CAMERA_ENABLE_CLIPPING_PLANES);
    clipPlanes = enableClippingPlanes ? getParamData(CAMERA_PROPERTY_CLIPPING_PLANES, nullptr) : nullptr;
    useHardwareRandomizer = getParam(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER.name.c_str(),
                                     static_cast<int>(DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER));

    dir = normalize(dir);
    ::ospray::vec3f dir_du = normalize(cross(dir, up));
    ::ospray::vec3f dir_dv = cross(dir_du, dir);

    ::ospray::vec3f org = pos;
    double imgPlane_size_y = 2.f * tanf(::ospray::deg2rad(0.5f * fieldOfView));
    double imgPlane_size_x = imgPlane_size_y * aspect;

    dir_du *= imgPlane_size_x;
    dir_dv *= imgPlane_size_y;

    ::ospray::vec3f dir_00 = dir - 0.5f * dir_du - 0.5f * dir_dv;

    double scaledAperture = 0.f;
    if (apertureRadius > 0.f)
    {
        dir_du *= focalDistance;
        dir_dv *= focalDistance;
        dir_00 *= focalDistance;
        scaledAperture = apertureRadius / imgPlane_size_x;
    }

    const auto clipPlaneData = clipPlanes ? clipPlanes->data : nullptr;
    const size_t numClipPlanes = clipPlanes ? clipPlanes->numItems : 0;

    ::ispc::AnaglyphCamera_set(getIE(), (const ::ispc::vec3f&)org, (const ::ispc::vec3f&)dir_00,
                               (const ::ispc::vec3f&)dir_du, (const ::ispc::vec3f&)dir_dv, scaledAperture, aspect,
                               (const ::ispc::vec4f*)clipPlaneData, numClipPlanes, nearClip, useHardwareRandomizer);
}

OSP_REGISTER_CAMERA(AnaglyphCamera, anaglyph);
} // namespace ospray
} // namespace engine
} // namespace core