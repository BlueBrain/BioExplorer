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

#include "AbstractManipulator.h"

namespace core
{
/**
 * Defines a flying mode camera manipulator, like in a flight simulator.
 */
class FlyingModeManipulator : public AbstractManipulator
{
public:
    FlyingModeManipulator(Camera& camera, KeyboardHandler& keyboardHandler);
    ~FlyingModeManipulator();

private:
    void dragLeft(const Vector2i& to, const Vector2i& from) final;
    void dragRight(const Vector2i& to, const Vector2i& from) final;
    void dragMiddle(const Vector2i& to, const Vector2i& from) final;
    void wheel(const Vector2i& position, float delta) final;

    void _strafeLeft();
    void _strafeRight();
    void _flyForward();
    void _flyBackwards();
};
} // namespace core
