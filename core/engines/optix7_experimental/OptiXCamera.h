/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include "OptiXContext.h"
#include "OptiXTypes.h"

#include <core/brayns/engineapi/Camera.h>

namespace brayns
{
/**
   OptiX specific camera

   This object is the OptiX specific implementation of a Camera
*/
class OptiXCamera : public Camera
{
public:
    OptiXCamera();
    ~OptiXCamera();

    /**
       Commits the changes held by the camera object so that
       attributes become available to the OptiX rendering engine
    */
    void commit();

protected:
    void _commitToOptiX();

#if 0
    sutil::Camera _camera{nullptr};
    optix::Buffer _clipPlanesBuffer{nullptr};
#endif
    Planes _clipPlanes;
    std::string _currentCamera;
    Vector3f _u, _v, _w;
    CUdeviceptr _d_miss_record;
};

} // namespace brayns
