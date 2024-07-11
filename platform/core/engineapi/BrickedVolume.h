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

#include <platform/core/engineapi/Volume.h>

namespace core
{
/**
 * @brief A volume type where the voxels are copied for each added brick.
 * @extends Volume
 */
class BrickedVolume : public virtual Volume
{
public:
    /**
     * @brief Sets a brick of data in the volume.
     * @param data The data to be set as a void pointer.
     * @param position The position of the brick as a Vector3ui object.
     * @param size The size of the brick as a Vector3ui object.
     */
    PLATFORM_API virtual void setBrick(const void* data, const Vector3ui& position, const Vector3ui& size) = 0;

protected:
    /**
     * @brief Constructs a new BrickedVolume object.
     * @param dimensions The dimensions of the volume as a Vector3ui object.
     * @param spacing The spacing between voxels as a Vector3f object.
     * @param type The data type of the volume.
     */
    BrickedVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType type)
        : Volume(dimensions, spacing, type)
    {
    }
};
} // namespace core
