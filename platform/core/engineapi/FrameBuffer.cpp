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

    // Create an OIIO ImageBuf and set the image data
    ImageSpec spec(width, height, channels, TypeDesc::UINT8);
    ImageBuf imageBuf(spec);
    imageBuf.set_pixels(spec.roi(), TypeDesc::UINT8, &imageData[0]);

    return imageBuf;
}
} // namespace core
