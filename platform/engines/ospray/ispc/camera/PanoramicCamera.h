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

#pragma once

#include "camera/Camera.h"

#include <platform/engines/ospray/OSPRayProperties.h>

namespace core
{
namespace engine
{
namespace ospray
{
struct OSPRAY_SDK_INTERFACE PanoramicCamera : public ::ospray::Camera
{
    PanoramicCamera();

    virtual std::string toString() const { return OSPRAY_CAMERA_PROPERTY_TYPE_PANORAMIC; }
    virtual void commit();

public:
    bool stereo{false};
    float interpupillaryDistance;
    bool enableClippingPlanes{false};
    ::ospray::Ref<::ospray::Data> clipPlanes;
    bool half{false};
};
} // namespace ospray
} // namespace engine
} // namespace core