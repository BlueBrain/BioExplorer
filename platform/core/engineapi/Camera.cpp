/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
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
