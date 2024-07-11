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

#include <platform/core/common/CommonTypes.h>
#include <platform/core/engineapi/Volume.h>

namespace core
{
/**
 * @class SharedDataVolume
 * @extends Volume
 * A volume type where the voxels are set once and only referenced from the source location.
 * Inherits from virtual Volume class.
 */
class SharedDataVolume : public virtual Volume
{
public:
    /**
     * @brief Sets the voxels of the volume.
     * @param voxels A pointer to the location of the voxels to be set.
     */
    PLATFORM_API virtual void setVoxels(const void* voxels) = 0;

    /**
     * @brief Convenience function to map data from file.
     * @param filename The file path to the data.
     */
    PLATFORM_API void mapData(const std::string& filename);

    /**
     * @brief Convenience function to map data from a buffer.
     * @param buffer The buffer containing the data.
     */
    PLATFORM_API void mapData(const uint8_ts& buffer);

    /**
     * @brief Convenience function to map data from a movable buffer.
     * @param buffer The movable buffer containing the data.
     */
    PLATFORM_API void mapData(uint8_ts&& buffer);

    /**
     * @brief Get the Memory Buffer object
     *
     * @return const uint8_ts& Memory buffer
     */
    const uint8_ts& getMemoryBuffer() const { return _memoryBuffer; }

protected:
    /**
     * @brief Constructs a new SharedDataVolume object.
     * @param dimensions The dimensions of the volume as a Vector3ui object.
     * @param spacing The spacing between voxels as a Vector3f object.
     * @param type The data type of the volume.
     */
    SharedDataVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType type)
        : Volume(dimensions, spacing, type)
    {
    }

    /**
     * @brief Destructs the SharedDataVolume object.
     * Unmaps the data from memory and closes the mapped file.
     */
    ~SharedDataVolume();

protected:
    uint8_ts _memoryBuffer; // The buffer containing the mapped data

private:
    void* _memoryMapPtr{nullptr}; // The pointer to the mapped memory
    int _cacheFileDescriptor{-1}; // The file descriptor of the mapped file
    size_t _size{0};              // The size of the mapped data
};
} // namespace core
