/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
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
#include <vector>

namespace core
{
enum class TextureWrapMode
{
    clamp_to_border,
    clamp_to_edge,
    mirror,
    repeat
};

class Texture2D
{
public:
    /**
     * Enum defining the types of textures.
     */
    enum class Type
    {
        default_,         // Default type of texture.
        cubemap,          // Cubemap texture.
        normal_roughness, // Normal roughness texture.
        aoe               // Ambient occlusion texture.
    };

    /**
     * Constructor for Texture2D class.
     *
     * @param type          Type of texture.
     * @param filename      File name of the texture.
     * @param channels      Number of channels in the texture.
     * @param depth         Depth of the texture.
     * @param width         Width of the texture.
     * @param height        Height of the texture.
     */
    PLATFORM_API Texture2D(const Type type, const std::string& filename, const uint8_t channels, const uint8_t depth,
                           const uint32_t width, const uint32_t height);

    /**
     * @return The size in bytes of the texture.
     */
    size_t getSizeInBytes() const { return height * width * depth * channels; }

    /**
     * Set the number of mip levels for the texture.
     *
     * @param mips      Number of mip levels to set.
     */
    void setMipLevels(const uint8_t mips);

    /**
     * @return The number of mip levels.
     */
    uint8_t getMipLevels() const { return _mipLevels; }

    /**
     * Get the raw data of the texture.
     *
     * @param face      Face of the texture.
     * @param mip       Mip level of the texture.
     * @return          The raw data of the texture.
     */
    template <typename T>
    const T* getRawData(const uint8_t face = 0, const uint8_t mip = 0) const
    {
        return reinterpret_cast<const T*>(_rawData[face][mip].data());
    }

    /**
     * Set the raw data of the texture.
     *
     * @param data      Raw data of the texture.
     * @param size      Size of the raw data.
     * @param face      Face of the texture.
     * @param mip       Mip level of the texture.
     */
    void setRawData(unsigned char* data, const size_t size, const uint8_t face = 0, const uint8_t mip = 0);

    /**
     * Set the raw data of the texture.
     *
     * @param rawData   Raw data of the texture.
     * @param face      Face of the texture.
     * @param mip       Mip level of the texture.
     */
    void setRawData(std::vector<unsigned char>&& rawData, const uint8_t face = 0, const uint8_t mip = 0);

    /**
     * @return The possible number of mip maps levels.
     */
    uint8_t getPossibleMipMapsLevels() const;

    /**
     * @return True if the texture is a cubemap.
     */
    bool isCubeMap() const { return type == Type::cubemap; }

    /**
     * @return True if the texture is a normal map.
     */
    bool isNormalMap() const { return type == Type::normal_roughness; }

    /**
     * @return The number of faces of the texture.
     */
    uint8_t getNumFaces() const { return isCubeMap() ? 6 : 1; }

    /**
     * Set the wrap mode of the texture.
     *
     * @param mode      Wrap mode to set.
     */
    void setWrapMode(const TextureWrapMode mode) { _wrapMode = mode; }

    /**
     * @return The wrap mode of the texture.
     */
    TextureWrapMode getWrapMode() const { return _wrapMode; }

    /** Type of the texture. */
    const Type type;

    /** File name of the texture. */
    const std::string filename;

    /** Number of channels in the texture. */
    const uint8_t channels;

    /** Depth of the texture. */
    const uint8_t depth;

    /** Width of the texture. */
    const uint32_t width;

    /** Height of the texture. */
    const uint32_t height;

private:
    /** Number of mip levels. */
    uint8_t _mipLevels{0};

    /** Wrap mode of the texture. */
    TextureWrapMode _wrapMode{TextureWrapMode::repeat};

    /**
     * The raw data of the texture.
     *
     * (faces, mips, pixels)
     */
    std::vector<std::vector<std::vector<unsigned char>>> _rawData;
};
} // namespace core
