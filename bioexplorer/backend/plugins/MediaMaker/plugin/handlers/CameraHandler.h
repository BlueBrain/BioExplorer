/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
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
