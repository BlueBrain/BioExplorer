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

#include "OSPRayCamera.h"
#include "OSPRayProperties.h"
#include "OSPRayRenderer.h"
#include "OSPRayUtils.h"

#include <platform/core/common/Properties.h>
#include <platform/core/engineapi/Scene.h>

namespace core
{
namespace engine
{
namespace ospray
{
OSPRayCamera::~OSPRayCamera()
{
    ospRelease(_camera);
}

void OSPRayCamera::commit()
{
    if (isModified())
    {
        const bool cameraChanged = _currentOSPCamera != getCurrentType();
        if (cameraChanged)
            _createOSPCamera();

        const auto position = getPosition();
        const auto dir = glm::rotate(getOrientation(), Vector3d(0., 0., -1.));
        const auto up = glm::rotate(getOrientation(), Vector3d(0., 1., 0.));

        osphelper::set(_camera, CAMERA_PROPERTY_POSITION, Vector3f(position));
        osphelper::set(_camera, CAMERA_PROPERTY_DIRECTION, Vector3f(dir));
        osphelper::set(_camera, CAMERA_PROPERTY_UP_VECTOR, Vector3f(up));
        osphelper::set(_camera, CAMERA_PROPERTY_BUFFER_TARGET, getBufferTarget());

        toOSPRayProperties(*this, _camera);

        // Clip planes
        if (!_clipPlanes.empty())
        {
            const auto clipPlanes = convertVectorToFloat(_clipPlanes);
            auto clipPlaneData = ospNewData(clipPlanes.size(), OSP_FLOAT4, clipPlanes.data());
            ospSetData(_camera, CAMERA_PROPERTY_CLIPPING_PLANES, clipPlaneData);
            ospRelease(clipPlaneData);
        }
        else
        {
            // ospRemoveParam leaks objects, so we set it to null first
            ospSetData(_camera, CAMERA_PROPERTY_CLIPPING_PLANES, nullptr);
            ospRemoveParam(_camera, CAMERA_PROPERTY_CLIPPING_PLANES);
        }
    }
    ospCommit(_camera);
}

void OSPRayCamera::setEnvironmentMap(const bool environmentMap)
{
    osphelper::set(_camera, CAMERA_PROPERTY_ENVIRONMENT_MAP, environmentMap);
    ospCommit(_camera);
}

void OSPRayCamera::setClipPlanes(const Planes& planes)
{
    if (_clipPlanes == planes)
        return;
    _clipPlanes = planes;
    markModified(false);
}

void OSPRayCamera::_createOSPCamera()
{
    auto newCamera = ospNewCamera(getCurrentType().c_str());
    if (!newCamera)
        throw std::runtime_error(getCurrentType() + " is not a registered camera");
    if (_camera)
        ospRelease(_camera);
    _camera = newCamera;
    _currentOSPCamera = getCurrentType();
    markModified(false);
}
} // namespace ospray
} // namespace engine
} // namespace core