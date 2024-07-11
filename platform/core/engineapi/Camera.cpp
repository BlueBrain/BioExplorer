/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include <platform/core/engineapi/Camera.h>

namespace core
{
Camera& Camera::operator=(const Camera& rhs)
{
    if (this == &rhs)
        return *this;

    clonePropertiesFrom(rhs);

    setPosition(rhs.getPosition());
    setOrientation(rhs.getOrientation());

    _initialPosition = rhs._initialPosition;
    _initialOrientation = rhs._initialOrientation;

    return *this;
}

void Camera::set(const Vector3d& position, const Quaterniond& orientation, const Vector3d& target)
{
    setPosition(position);
    setOrientation(orientation);
    setTarget(target);
}

void Camera::setInitialState(const Vector3d& position, const Quaterniond& orientation, const Vector3d& target)
{
    _initialPosition = position;
    _initialTarget = target;
    _initialOrientation = orientation;
    _initialOrientation = glm::normalize(_initialOrientation);
    set(position, orientation, target);
}

void Camera::reset()
{
    set(_initialPosition, _initialOrientation, _initialTarget);
}

std::ostream& operator<<(std::ostream& os, Camera& camera)
{
    const auto& position = camera.getPosition();
    const auto& orientation = camera.getOrientation();
    return os << position << ", " << orientation;
}
} // namespace core
