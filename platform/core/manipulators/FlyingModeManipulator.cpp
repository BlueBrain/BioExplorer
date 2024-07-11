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

#include "FlyingModeManipulator.h"

#include <platform/core/engineapi/Camera.h>

#include <platform/core/common/Logs.h>
#include <platform/core/common/input/KeyboardHandler.h>

namespace core
{
FlyingModeManipulator::FlyingModeManipulator(Camera& camera, KeyboardHandler& keyboardHandler)
    : AbstractManipulator(camera, keyboardHandler)
{
    _keyboardHandler.registerKeyboardShortcut('a', "Strafe left", std::bind(&FlyingModeManipulator::_strafeLeft, this));
    _keyboardHandler.registerKeyboardShortcut('d', "Strafe right",
                                              std::bind(&FlyingModeManipulator::_strafeRight, this));
    _keyboardHandler.registerKeyboardShortcut('w', "Fly forward", std::bind(&FlyingModeManipulator::_flyForward, this));
    _keyboardHandler.registerKeyboardShortcut('s', "Fly backwards",
                                              std::bind(&FlyingModeManipulator::_flyBackwards, this));
}

FlyingModeManipulator::~FlyingModeManipulator()
{
    _keyboardHandler.unregisterKeyboardShortcut('a');
    _keyboardHandler.unregisterKeyboardShortcut('d');
    _keyboardHandler.unregisterKeyboardShortcut('w');
    _keyboardHandler.unregisterKeyboardShortcut('s');
}

void FlyingModeManipulator::dragLeft(const Vector2i& to, const Vector2i& from)
{
    const float du = (to.x - from.x) * getRotationSpeed();
    const float dv = (to.y - from.y) * getRotationSpeed();
    rotate(_camera.getPosition(), du, dv, AxisMode::globalY);
}

void FlyingModeManipulator::dragRight(const Vector2i& to, const Vector2i& from)
{
    const float distance = -(to.y - from.y) * DEFAULT_MOUSE_MOTION_SPEED_MULTIPLIER * getMotionSpeed();
    translate(Vector3f(0, 0, -1) * distance);
}

void FlyingModeManipulator::dragMiddle(const Vector2i& to, const Vector2i& from)
{
    const float x = (to.x - from.x) * DEFAULT_MOUSE_MOTION_SPEED_MULTIPLIER * getMotionSpeed();
    const float y = (to.y - from.y) * DEFAULT_MOUSE_MOTION_SPEED_MULTIPLIER * getMotionSpeed();
    translate({-x, y, 0.f});
}

void FlyingModeManipulator::wheel(const Vector2i& /*position*/, const float delta)
{
    translate(Vector3f(0, 0, -1) * delta * getWheelSpeed());
}

void FlyingModeManipulator::_strafeLeft()
{
    translate(Vector3f(-1, 0, 0) * getMotionSpeed());
}

void FlyingModeManipulator::_strafeRight()
{
    translate(Vector3f(1, 0, 0) * getMotionSpeed());
}

void FlyingModeManipulator::_flyForward()
{
    translate(Vector3f(0, 0, -1) * getMotionSpeed());
}

void FlyingModeManipulator::_flyBackwards()
{
    translate(Vector3f(0, 0, 1) * getMotionSpeed());
}
} // namespace core
