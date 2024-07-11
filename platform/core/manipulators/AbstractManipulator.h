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

#include <platform/core/common/Types.h>

constexpr float DEFAULT_MOUSE_MOTION_SPEED_MULTIPLIER = 0.25f;

namespace core
{
/**
 * Base class for camera manipulators.
 */
class AbstractManipulator
{
public:
    enum class AxisMode
    {
        globalY = 0,
        localY
    };

    AbstractManipulator(Camera& camera, KeyboardHandler& keyboardHandler);
    virtual ~AbstractManipulator() = default;

    /** Adjust manipulator behaviour to the given scene */
    virtual void adjust(const Boxd& boundingBox);
    virtual void dragLeft(const Vector2i& to, const Vector2i& from) = 0;
    virtual void dragRight(const Vector2i& to, const Vector2i& from) = 0;
    virtual void dragMiddle(const Vector2i& to, const Vector2i& from) = 0;
    virtual void wheel(const Vector2i& position, float delta) = 0;

    float getMotionSpeed() const;
    void updateMotionSpeed(float speed);

    float getRotationSpeed() const;
    float getWheelSpeed() const;
    void rotate(const Vector3d& pivot, double du, double dv, AxisMode axisMode);

protected:
    /*! target camera */
    Camera& _camera;

    /*! keyboard handler to register/deregister keyboard events */
    KeyboardHandler& _keyboardHandler;

    /*! camera speed modifier - affects how many units the camera _moves_ with
     * each unit on the screen */
    double _motionSpeed;

    /*! camera rotation speed modifier - affects how many units the camera
     * _rotates_ with each unit on the screen */
    double _rotationSpeed;

    void translate(const Vector3d& v);
};
} // namespace core
