/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

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

#include "CylindricStereoTrackedCamera.h"
#include "CylindricStereoTrackedCamera_ispc.h"

#include <platform/core/common/Properties.h>

using namespace core;

namespace ospray
{
namespace
{
constexpr uint8_t leftWall = 0u;
constexpr uint8_t rightWall = 1u;
constexpr uint8_t leftFloor = 2u;
constexpr uint8_t rightFloor = 3u;

const vec3f OPENDECK_RIGHT_DIRECTION{1.0f, 0.0f, 0.0f};

vec3f _rotateVectorByQuat(const vec3f& v, const vec4f& q)
{
    const auto u = vec3f{q[0], q[1], q[2]}; // vector part of the quaternion
    const auto s = q[3];                    // scalar part of the quaternion

    return 2.0f * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}
} // namespace

CylindricStereoTrackedCamera::CylindricStereoTrackedCamera()
{
    ispcEquivalent = ::ispc::CylindricStereoTrackedCamera_create(this);
}

std::string CylindricStereoTrackedCamera::toString() const
{
    return "ospray::CylindricStereoTrackedCamera";
}

void CylindricStereoTrackedCamera::commit()
{
    Camera::commit();

    const std::string& bufferTarget = getParamString(CAMERA_PROPERTY_BUFFER_TARGET);
    const float cameraScaling = getParamf("cameraScaling", 1.0);

    uint8_t bufferId = 255u;
    if (bufferTarget == "0L")
        bufferId = leftWall;
    else if (bufferTarget == "0R")
        bufferId = rightWall;
    else if (bufferTarget == "1L")
        bufferId = leftFloor;
    else if (bufferTarget == "1R")
        bufferId = rightFloor;

    const auto stereoMode = getStereoMode();
    const auto ipd = getInterpupillaryDistance(stereoMode);

    const auto headPosition = _getHeadPosition();

    // The tracking model of the 3d glasses is inverted so we need to negate CamDu here.
    const auto openDeckCamDU = -_getOpendeckCamDU();

    dir = vec3f(0, 0, 1);
    const auto org = pos;
    const auto dir_du = vec3f(1, 0, 0);
    const auto dir_dv = vec3f(0, 1, 0);

    ::ispc::CylindricStereoTrackedCamera_set(getIE(), (const ::ispc::vec3f&)org, (const ::ispc::vec3f&)headPosition,
                                             (const ::ispc::vec3f&)dir, (const ::ispc::vec3f&)dir_du,
                                             (const ::ispc::vec3f&)dir_dv, (const ::ispc::vec3f&)openDeckCamDU, ipd,
                                             bufferId, cameraScaling);
}

vec3f CylindricStereoTrackedCamera::_getHeadPosition()
{
    return getParam3f("headPosition", vec3f(0.0f, 2.0f, 0.0f));
}

vec3f CylindricStereoTrackedCamera::_getOpendeckCamDU()
{
    const auto quat = getParam4f("headRotation", vec4f(0.0f, 0.0f, 0.0f, 1.0f));
    const auto cameraDU = _rotateVectorByQuat(OPENDECK_RIGHT_DIRECTION, quat);
    return normalize(cameraDU);
}

OSP_REGISTER_CAMERA(CylindricStereoTrackedCamera, cylindricStereoTracked);
} // namespace ospray
