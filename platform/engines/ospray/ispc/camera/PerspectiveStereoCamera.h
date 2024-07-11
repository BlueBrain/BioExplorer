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

#pragma once

#include "camera/Camera.h"

#include <platform/core/common/CommonTypes.h>
#include <platform/core/common/Properties.h>

namespace core
{
namespace engine
{
namespace ospray
{
struct OSPRAY_SDK_INTERFACE PerspectiveStereoCamera : public ::ospray::Camera
{
    /**
     * @brief Construct a new Perspective Stereo Camera object
     *
     */
    PerspectiveStereoCamera();

    /**
     * @brief Returns the name of the camera
     *
     * @return std::string The name of the camera
     */
    virtual std::string toString() const { return CAMERA_PROPERTY_TYPE_PERSPECTIVE; }

    /**
     * @brief Commit the changes to the OSPRay engine
     *
     */
    virtual void commit();

public:
    double fieldOfView;
    double aspect;
    double apertureRadius;
    double focalDistance;
    bool architectural;

    // Clip planes
    bool enableClippingPlanes{false};
    ::ospray::Ref<::ospray::Data> clipPlanes;

    // Stereo
    CameraStereoMode stereoMode;
    double interpupillaryDistance;

    bool useHardwareRandomizer{false};
};
} // namespace ospray
} // namespace engine
} // namespace core