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
