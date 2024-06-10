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
