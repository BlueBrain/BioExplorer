/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Juan Hernando <juan.hernando@epfl.ch>
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
#include <platform/core/common/Types.h>

SERIALIZATION_ACCESS(ClipPlane)

namespace core
{
class ClipPlane : public BaseObject
{
public:
    /**
     * @brief Constructor.
     * @param plane A constant reference to Plane object.
     */
    ClipPlane(const Plane& plane)
        : _id(_nextID++)
        , _plane(plane)
    {
    }

    /**
     * @brief Default constructor defined internally.
     */
    ClipPlane() = default;

    /**
     * @brief Constructor with id and plane defined internally.
     * @param id A size_t representing the ID of this clip plane object.
     * @param plane A constant reference to the Plane object.
     */
    ClipPlane(const size_t id, const Plane& plane);

    /**
     * @brief Returns id of this clip plane object.
     * @return size_t The id of the clip plane object.
     */
    size_t getID() const { return _id; }

    /**
     * @brief Returns the constant reference to the Plane object of this clip plane object.
     * @return const Plane& A reference to the constant Plane object of this clip plane object.
     */
    const Plane& getPlane() const { return _plane; };

    /**
     * @brief Sets the Plane object of this clip plane object.
     * @param plane A constant reference to the Plane object to be set.
     */
    void setPlane(const Plane& plane) { _updateValue(_plane, plane); }

private:
    static size_t _nextID; //!< A static variable to get next ID for a clip plane object.
    size_t _id = 0;        //!< The ID of this clip plane object.
    Plane _plane = {{0}};  //!< The Plane object of this clip plane object.

    /**
     * @brief A friend function to help with serialization of ClipPlane objects.
     */
    SERIALIZATION_FRIEND(ClipPlane);
};
} // namespace core
