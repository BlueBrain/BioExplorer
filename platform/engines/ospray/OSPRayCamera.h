/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include "OSPRayEngine.h"

#include <platform/core/engineapi/Camera.h>

#include <ospray.h>

namespace core
{
namespace engine
{
namespace ospray
{
/**
   OPSRAY specific camera

   This object is the OSPRay specific implementation of a Camera
*/
class OSPRayCamera : public Camera
{
public:
    OSPRayCamera(OSPRayEngine* engine) { _engine = engine; }

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
} // namespace ospray
} // namespace engine
} // namespace core
