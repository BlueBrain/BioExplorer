/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Daniel Nachbaur <daniel.nachbaur@epfl.ch>
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
