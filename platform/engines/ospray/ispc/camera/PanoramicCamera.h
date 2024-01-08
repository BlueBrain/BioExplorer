/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Grigori Chevtchenko <grigori.chevtchenko@epfl.ch>
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include "camera/Camera.h"

#include <platform/engines/ospray/OSPRayProperties.h>

namespace core
{
namespace engine
{
namespace ospray
{
struct OSPRAY_SDK_INTERFACE PanoramicCamera : public ::ospray::Camera
{
    PanoramicCamera();

    virtual std::string toString() const { return OSPRAY_CAMERA_PROPERTY_TYPE_PANORAMIC; }
    virtual void commit();

public:
    bool stereo{false};
    float interpupillaryDistance;
    bool enableClippingPlanes{false};
    ::ospray::Ref<::ospray::Data> clipPlanes;
    bool half{false};
};
} // namespace ospray
} // namespace engine
} // namespace core