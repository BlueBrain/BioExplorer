/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *                     Jafet Villafranca <jafet.villafrancadiaz@epfl.ch>
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

#pragma once

#include <platform/core/common/Api.h>
#include <platform/core/common/PropertyObject.h>
#include <platform/core/common/Types.h>

SERIALIZATION_ACCESS(Camera)

namespace core
{
/**
 * @class Camera
 * @extends PropertyObject
 * @brief The Camera class is an abstract interface for a camera in a 3D graphics application.
 * It is defined by a position and a quaternion and inherits from the PropertyObject class.
 */
class Camera : public PropertyObject
{
public:
    /**
     * @brief Default constructor.
     */
    PLATFORM_API Camera() = default;

    /**
     * @brief Default destructor.
     */
    PLATFORM_API virtual ~Camera() = default;

    /**
     * @brief Commits any changes made to the camera object so that
     * attributes become available to the rendering engine
     */
    PLATFORM_API virtual void commit(){};

    /**
     * @brief Copy constructor.
     */
    PLATFORM_API Camera& operator=(const Camera& rhs);

    /**
     * @brief Sets the position, orientation quaternion, and target of the camera.
     *
     * @param position The x, y, z coordinates of the camera position.
     * @param orientation The x, y, z, w values of the quaternion describing the camera orientation.
     * @param target The x, y, z coordinates of the camera target.
     */
    PLATFORM_API void set(const Vector3d& position, const Quaterniond& orientation,
                          const Vector3d& target = Vector3d(0.0, 0.0, 0.0));

    /**
     * @brief Sets the initial state of the camera.
     *
     * @param position The x, y, z coordinates of the camera position.
     * @param orientation The x, y, z, w values of the quaternion describing the camera orientation.
     * @param target The x, y, z coordinates of the camera target.
     */
    PLATFORM_API void setInitialState(const Vector3d& position, const Quaterniond& orientation,
                                      const Vector3d& target = Vector3d(0.0, 0.0, 0.0));

    /**
     * @brief Resets the camera to its initial values.
     */
    PLATFORM_API void reset();

    /**
     * @brief Sets the camera position.
     *
     * @param position The x, y, z coordinates of the camera position.
     */
    PLATFORM_API void setPosition(const Vector3d& position) { _updateValue(_position, position); }

    /**
     * @brief Sets the camera target.
     *
     * @param target The x, y, z coordinates of the camera target.
     */
    PLATFORM_API void setTarget(const Vector3d& target) { _updateValue(_target, target); }

    /**
     * @brief Gets the camera position.
     *
     * @return The x, y, z coordinates of the camera position.
     */
    PLATFORM_API const Vector3d& getPosition() const { return _position; }

    /**
     * @brief Gets the camera target.
     *
     * @return The x, y, z coordinates of the camera target.
     */
    PLATFORM_API const Vector3d& getTarget() const { return _target; }

    /**
     * @brief Sets the camera orientation quaternion.
     *
     * @param orientation The orientation quaternion.
     */
    PLATFORM_API void setOrientation(Quaterniond orientation)
    {
        orientation = glm::normalize(orientation);
        _updateValue(_orientation, orientation);
    }

    /**
     * @brief Gets the camera orientation quaternion.
     *
     * @return The orientation quaternion.
     */
    PLATFORM_API const Quaterniond& getOrientation() const { return _orientation; }

    /**
     * @brief Sets the name of the current rendered frame buffer.
     *
     * @param target The name of the frame buffer.
     */
    PLATFORM_API void setBufferTarget(const std::string& target) { _updateValue(_bufferTarget, target, false); }

    /**
     * @brief Gets the name of the current rendered frame buffer.
     *
     * @return The name of the frame buffer.
     */
    PLATFORM_API const std::string& getBufferTarget() const { return _bufferTarget; }

private:
    Vector3d _target;
    Vector3d _position;
    Quaterniond _orientation;

    Vector3d _initialTarget;
    Vector3d _initialPosition;
    Quaterniond _initialOrientation;

    std::string _bufferTarget;

    SERIALIZATION_FRIEND(Camera);
};

/**
 * @brief Overloads the << operator for a Camera object.
 *
 * @param os The output stream.
 * @param camera The camera object to output.
 * @return The output stream.
 */
std::ostream& operator<<(std::ostream& os, Camera& camera);
} // namespace core
