/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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