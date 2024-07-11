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

#include <Defines.h>

#include <platform/core/common/Types.h>

#include <vector>

namespace bioexplorer
{
namespace mediamaker
{
enum class FrameBufferMode
{
    color = 0,
    depth = 1
};

typedef struct
{
    core::Vector3d origin;
    core::Vector3d direction;
    core::Vector3d up;
    double apertureRadius;
    double focalDistance;
    double interpupillaryDistance;
} CameraKeyFrame;

using CameraKeyFrames = std::vector<CameraKeyFrame>;

} // namespace mediamaker
} // namespace bioexplorer