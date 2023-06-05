/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#ifndef OSPRAYCAMERA_H
#define OSPRAYCAMERA_H

#include <platform/core/engineapi/Camera.h>
#include <ospray.h>

namespace core
{
/**
   OPSRAY specific camera

   This object is the OSPRay specific implementation of a Camera
*/
class OSPRayCamera : public Camera
{
public:
    OSPRayCamera() = default;
    ~OSPRayCamera();

    /**
       Commits the changes held by the camera object so that
       attributes become available to the OSPRay rendering engine
    */
    void commit() final;

    /**
       Set the clipping planes to use in this camera.
       Only implemented in the perspective and orthographic cameras.
    */
    void setClipPlanes(const Planes& planes);

    void setEnvironmentMap(const bool environmentMap);

    /**
       Gets the OSPRay implementation of the camera object
       @return OSPRay implementation of the camera object
    */
    OSPCamera impl() { return _camera; }

private:
    OSPCamera _camera{nullptr};
    std::string _currentOSPCamera;
    Planes _clipPlanes;

    void _createOSPCamera();
};
} // namespace core
#endif // OSPRAYCAMERA_H
