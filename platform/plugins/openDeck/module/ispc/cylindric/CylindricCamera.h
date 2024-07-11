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

#include <ospray/SDK/camera/Camera.h>

namespace ospray
{
/**
 * This camera is designed for the opendeck. It has a fixed
 * vertical field of view of 48.549 degrees. The rays are using
 * cylindrical projection for the x axis and perspective projection
 * for the y axis of an image.
 */
struct CylindricCamera : public Camera
{
    CylindricCamera();
    std::string toString() const override;
    void commit() override;
};
} // namespace ospray
