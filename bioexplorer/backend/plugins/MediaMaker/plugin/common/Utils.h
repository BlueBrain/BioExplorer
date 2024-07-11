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

#include "Types.h"

#include <plugin/api/Params.h>

#include <platform/core/common/Types.h>

namespace bioexplorer
{
namespace mediamaker
{

/**
 * @brief Convert a CameraDefinition structure into a cameraKeyFrame structure
 *
 * @param cameraDefinition CameraDefinition structure
 * @return CameraKeyFrame CameraKeyFrame structure
 */
CameraKeyFrame cameraDefinitionToKeyFrame(const CameraDefinition &cameraDefinition);

/**
 * @brief Convert a cameraKeyFrame structure into a CameraDefinition structure
 *
 * @param camera Camera object
 * @return CameraDefinition CameraDefinition structure
 */
CameraDefinition keyFrameToCameraDefinition(const CameraKeyFrame &camera);

/**
 * @brief Set the Camera object
 *
 * @param cameraKeyFrame Camera key frame
 * @param camera Camera object
 * @param triggerCallback Triggers call back over websocket if enabled
 */
void setCamera(const CameraKeyFrame &cameraKeyFrame, core::Camera &camera, const bool triggerCallback = true);

/**
 * @brief Get the Camera Key Frame object
 *
 * @param camera Camera object
 * @return CameraKeyFrame CameraKeyFrame structure
 */
CameraKeyFrame getCameraKeyFrame(core::Camera &camera);

} // namespace mediamaker
} // namespace bioexplorer
