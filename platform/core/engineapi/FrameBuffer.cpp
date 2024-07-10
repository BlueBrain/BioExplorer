/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include "FrameBuffer.h"

OIIO_NAMESPACE_USING

namespace core
{
FrameBuffer::FrameBuffer(const std::string& name, const Vector2ui& frameSize, const FrameBufferFormat frameBufferFormat)
    : _name(name)
    , _frameSize(frameSize)
    , _frameBufferFormat(frameBufferFormat)
{
}

size_t FrameBuffer::getColorDepth() const
{
    switch (_frameBufferFormat)
    {
    case FrameBufferFormat::rgba_i8:
    case FrameBufferFormat::bgra_i8:
    case FrameBufferFormat::rgb_f32:
        return 4;
    case FrameBufferFormat::rgb_i8:
        return 3;
    default:
        return 0;
    }
}

ImageBuf FrameBuffer::getImage()
{
    map();
    const auto colorBuffer = getColorBuffer();
    const auto& size = getSize();
    const unsigned int width = size.x;
    const unsigned int height = size.y;
    const unsigned int depth = getColorDepth();
    const unsigned int channels = 4; // Assuming RGBA

    std::vector<unsigned char> imageData(colorBuffer, colorBuffer + width * height * channels);

    unmap();

#if FREEIMAGE_COLORORDER == FREEIMAGE_COLORORDER_BGR
    // Swap red and blue channels
    for (unsigned int y = 0; y < height; ++y)
    {
        for (unsigned int x = 0; x < width; ++x)
        {
            unsigned int index = (y * width + x) * channels;
            std::swap(imageData[index], imageData[index + 2]); // Swap red and blue channels
        }
    }
#endif

    // Create an OIIO ImageBuf and set the image data
    ImageSpec spec(width, height, channels, TypeDesc::UINT8);
    ImageBuf imageBuf(spec);
    imageBuf.set_pixels(spec.roi(), TypeDesc::UINT8, &imageData[0]);

    return imageBuf;
}
} // namespace core
