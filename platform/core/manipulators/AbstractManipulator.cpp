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

#include "AbstractManipulator.h"

#include <platform/core/common/Logs.h>
#include <platform/core/common/input/KeyboardHandler.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Scene.h>

namespace core
{
namespace
{
constexpr float DEFAULT_MOTION_SPEED = 0.03f;
constexpr float DEFAULT_ROTATION_SPEED = 0.006f;
} // namespace

AbstractManipulator::AbstractManipulator(Camera& camera, KeyboardHandler& keyboardHandler)
    : _camera(camera)
    , _keyboardHandler(keyboardHandler)
    , _motionSpeed{DEFAULT_ROTATION_SPEED}
    , _rotationSpeed{DEFAULT_ROTATION_SPEED}
{
}

void AbstractManipulator::adjust(const Boxd& boundingBox)
{
    const auto size = boundingBox.isEmpty() ? 1 : glm::compMax(boundingBox.getSize());
    auto position = boundingBox.getCenter();
    auto target = position;
    position.z += size;

    _camera.setInitialState(position, glm::identity<Quaterniond>(), target);

    _motionSpeed = DEFAULT_MOTION_SPEED * size;

    if (boundingBox.isEmpty())
        CORE_INFO("World bounding box: empty")
    else
        CORE_INFO("World bounding box: " << boundingBox);
    CORE_INFO("World center      : " << boundingBox.getCenter());
}

float AbstractManipulator::getRotationSpeed() const
{
    return _rotationSpeed;
}

float AbstractManipulator::getWheelSpeed() const
{
    return getMotionSpeed() * 5.f;
}

float AbstractManipulator::getMotionSpeed() const
{
    return _motionSpeed;
}

void AbstractManipulator::updateMotionSpeed(const float speed)
{
    _motionSpeed *= speed;
}

void AbstractManipulator::translate(const Vector3d& vector)
{
    auto orientation = _camera.getOrientation();
    const auto translation = glm::rotate(orientation, vector);

    _camera.setPosition(_camera.getPosition() + translation);
}

void AbstractManipulator::rotate(const Vector3d& pivot, const double du, const double dv, AxisMode axisMode)
{
    const Vector3d axisX = glm::rotate(_camera.getOrientation(), Vector3d(1.0, 0.0, 0.0));

    const Vector3d axisY = axisMode == AxisMode::localY ? glm::rotate(_camera.getOrientation(), Vector3d(0.0, 1.0, 0.0))
                                                        : Vector3d(0.0, 1.0, 0.0);

    const Quaterniond deltaU = glm::angleAxis(-du, axisY);
    const Quaterniond deltaV = glm::angleAxis(-dv, axisX);

    const Quaterniond final = deltaU * deltaV * _camera.getOrientation();
    const Vector3d dir = glm::rotate(final, Vector3d(0.0, 0.0, -1.0));

    const double rotationRadius = glm::length(_camera.getPosition() - pivot);
    _camera.setPosition(pivot + rotationRadius * -dir);
    _camera.setOrientation(final);
    _camera.setTarget(pivot);
}
} // namespace core
