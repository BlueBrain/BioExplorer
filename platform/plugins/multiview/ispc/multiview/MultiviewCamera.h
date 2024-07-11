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

#pragma once

#include <ospray/SDK/camera/Camera.h>

namespace ospray
{
/**
 * This is a 4 view camera. You can see from the top,
 * right, front and perspective viewports.
 */
struct OSPRAY_SDK_INTERFACE MultiviewCamera : public Camera
{
    MultiviewCamera();
    std::string toString() const override;
    void commit() override;

    // Clip planes
    Ref<Data> clipPlanes;
};
} // namespace ospray
