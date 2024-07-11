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

#include <platform/engines/ospray/OSPRayProperties.h>

#include <ospray/SDK/camera/Camera.h>

namespace core
{
namespace engine
{
namespace ospray
{
/**
 * This camera is an extension of the regular ospray stereoscopic camera. It has an additional option to select the
 * distance of the zero-parallax plane.
 */
struct PerspectiveParallaxCamera : public ::ospray::Camera
{
    PerspectiveParallaxCamera();
    virtual ~PerspectiveParallaxCamera() override = default;

    virtual std::string toString() const override { return OSPRAY_CAMERA_PROPERTY_TYPE_PERSPECTIVE_PARALLAX; }
    virtual void commit() override;

    typedef enum
    {
        OSP_STEREO_NONE,
        OSP_STEREO_LEFT,
        OSP_STEREO_RIGHT,
        OSP_STEREO_SIDE_BY_SIDE
    } StereoMode;
};
} // namespace ospray
} // namespace engine
} // namespace core