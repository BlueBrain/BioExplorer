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

#include "camera/Camera.h"

#include <platform/core/common/CommonTypes.h>
#include <platform/core/common/Properties.h>

namespace core
{
namespace engine
{
namespace ospray
{
struct OSPRAY_SDK_INTERFACE AnaglyphCamera : public ::ospray::Camera
{
    /**
     * @brief Construct a new Perspective Stereo Camera object
     *
     */
    AnaglyphCamera();

    /**
     * @brief Returns the name of the camera
     *
     * @return std::string The name of the camera
     */
    virtual std::string toString() const { return CAMERA_PROPERTY_TYPE_PERSPECTIVE; }

    /**
     * @brief Commit the changes to the OSPRay engine
     *
     */
    virtual void commit();

public:
    double fieldOfView;
    double aspect;
    double apertureRadius;
    double focalDistance;
    bool architectural;

    // Clip planes
    bool enableClippingPlanes{false};
    ::ospray::Ref<::ospray::Data> clipPlanes;

    // Stereo
    double interpupillaryDistance;

    bool useHardwareRandomizer{false};
};
} // namespace ospray
} // namespace engine
} // namespace core