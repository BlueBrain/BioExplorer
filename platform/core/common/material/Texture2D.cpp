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

#include "Texture2D.h"

namespace core
{
Texture2D::Texture2D(const Type type_, const std::string& filename_, const uint8_t channels_, const uint8_t depth_,
                     const uint32_t width_, const uint32_t height_)
    : type(type_)
    , filename(filename_)
    , channels(channels_)
    , depth(depth_)
    , width(width_)
    , height(height_)
{
    _rawData.resize(type == Type::cubemap ? 6 : 1);
    setMipLevels(1);
}

void Texture2D::setMipLevels(const uint8_t mips)
{
    if (mips == _mipLevels)
        return;
    _mipLevels = mips;
    for (auto& data : _rawData)
        data.resize(mips);
}

void Texture2D::setRawData(unsigned char* data, const size_t size, const uint8_t face, const uint8_t mip)
{
    _rawData[face][mip].clear();
    _rawData[face][mip].assign(data, data + size);
}

void Texture2D::setRawData(std::vector<unsigned char>&& rawData, const uint8_t face, const uint8_t mip)
{
    _rawData[face][mip] = std::move(rawData);
}

uint8_t Texture2D::getPossibleMipMapsLevels() const
{
    uint8_t mipMapLevels = 1u;
    auto nx = width;
    auto ny = height;
    while (nx % 2 == 0 && ny % 2 == 0)
    {
        nx /= 2;
        ny /= 2;
        ++mipMapLevels;
    }
    return mipMapLevels;
}
} // namespace core
