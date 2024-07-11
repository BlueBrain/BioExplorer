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

#include <common/Properties.h>

#include <ospray/SDK/camera/Camera.h>

namespace sonataexplorer
{
struct SphereClippingPerspectiveCamera : ::ospray::Camera
{
    SphereClippingPerspectiveCamera();

    virtual std::string toString() const { return CAMERA_SPHERE_CLIPPING_PERSPECTIVE; }
    virtual void commit();

public:
    float fovy;
    float aspect;
    float apertureRadius;
    float focalDistance;
    bool architectural; // orient image plane to be parallel to 'up' and shift
                        // the lens
    bool stereo;
    float interpupillaryDistance; // distance between the two cameras (stereo)
    bool enableClippingPlanes{false};
    ::ospray::Ref<::ospray::Data> clipPlanes;
    bool useHardwareRandomizer{false};
};
} // namespace sonataexplorer
