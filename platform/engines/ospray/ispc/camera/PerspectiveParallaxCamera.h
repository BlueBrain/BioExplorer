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