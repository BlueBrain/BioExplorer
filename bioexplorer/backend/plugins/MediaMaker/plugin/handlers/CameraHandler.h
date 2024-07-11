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

#include <plugin/common/Types.h>

#include <platform/core/common/simulation/AbstractAnimationHandler.h>
#include <platform/core/engineapi/Camera.h>

namespace bioexplorer
{
namespace mediamaker
{
/**
 * @brief The CameraHandler handles the position of orientation of the camera from the given set of key frames,
 * smoothing paramters
 *
 */
class CameraHandler : public core::AbstractAnimationHandler
{
public:
    /** @copydoc AbstractAnimationHandler::AbstractAnimationHandler */
    CameraHandler(core::Camera& camera, const CameraKeyFrames& cameraInformation, const uint64_t stepsBetweenKeyFrames,
                  const uint64_t numberOfSmoothingSteps);

    /** @copydoc AbstractAnimationHandler::AbstractAnimationHandler */
    CameraHandler(const CameraHandler& rhs);

    /** @copydoc AbstractAnimationHandler::getFrameData */
    void* getFrameData(const uint32_t frame) final;

    /** @copydoc AbstractAnimationHandler::clone */
    core::AbstractSimulationHandlerPtr clone() const final;

private:
    void _logSimulationInformation();
    void _buildCameraPath();

    CameraKeyFrames _keyFrames;
    uint64_t _stepsBetweenKeyFrames;
    uint64_t _numberOfSmoothingSteps;
    core::Camera& _camera;

    CameraKeyFrames _smoothedKeyFrames;
};
} // namespace mediamaker
} // namespace bioexplorer
