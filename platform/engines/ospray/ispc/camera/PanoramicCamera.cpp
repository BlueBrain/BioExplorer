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