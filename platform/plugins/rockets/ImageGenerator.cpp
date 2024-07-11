/*
    Copyright 2015 - 2017 Blue Brain Project / EPFL

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

#include "ImageGenerator.h"

#include <platform/core/common/utils/ImageUtils.h>
#include <platform/core/common/utils/base64/base64.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/parameters/ApplicationParameters.h>

OIIO_NAMESPACE_USING

namespace core
{
const size_t DEFAULT_COLOR_DEPTH = 4;

ImageGenerator::~ImageGenerator()
{
    if (_compressor)
        tjDestroy(_compressor);
}

ImageGenerator::ImageBase64 ImageGenerator::createImage(FrameBuffer& frameBuffer, const std::string& format,
                                                        const uint8_t quality)
{
    return {getBase64Image(frameBuffer.getImage(), format, quality)};
}

ImageGenerator::ImageBase64 ImageGenerator::createImage(const std::vector<FrameBufferPtr>& frameBuffers,
                                                        const std::string& format, const uint8_t quality)
{
    if (frameBuffers.size() == 1)
        return {getBase64Image(frameBuffers[0]->getImage(), format, quality)};

    std::vector<ImageBuf> images;
    for (auto frameBuffer : frameBuffers)
        images.push_back(frameBuffer->getImage());
    return {getBase64Image(mergeImages(images), format, quality)};
}

ImageGenerator::ImageJPEG ImageGenerator::createJPEG(FrameBuffer& frameBuffer, const uint8_t quality)
{
    frameBuffer.map();
    const auto colorBuffer = frameBuffer.getColorBuffer();
    if (!colorBuffer)
    {
        frameBuffer.unmap();
        return ImageJPEG();
    }

    int32_t pixelFormat = TJPF_RGBX;
    switch (frameBuffer.getFrameBufferFormat())
    {
    case FrameBufferFormat::bgra_i8:
        pixelFormat = TJPF_BGRX;
        break;
    case FrameBufferFormat::rgba_i8:
    default:
        pixelFormat = TJPF_RGBX;
    }

    const auto& frameSize = frameBuffer.getSize();
    ImageJPEG image;
    image.data = _encodeJpeg(frameSize.x, frameSize.y, colorBuffer, pixelFormat, quality, image.size);
    frameBuffer.unmap();
    return image;
}

ImageGenerator::ImageJPEG::JpegData ImageGenerator::_encodeJpeg(const uint32_t width, const uint32_t height,
                                                                const uint8_t* rawData, const int32_t pixelFormat,
                                                                const uint8_t quality, unsigned long& dataSize)
{
    uint8_t* tjSrcBuffer = const_cast<uint8_t*>(rawData);
    const int32_t color_components = DEFAULT_COLOR_DEPTH;
    const int32_t tjPitch = width * color_components;
    const int32_t tjPixelFormat = pixelFormat;

    uint8_t* tjJpegBuf = 0;
    const int32_t tjJpegSubsamp = TJSAMP_444;
    const int32_t tjFlags = TJXOP_ROT180;

    const int32_t success = tjCompress2(_compressor, tjSrcBuffer, width, tjPitch, height, tjPixelFormat, &tjJpegBuf,
                                        &dataSize, tjJpegSubsamp, quality, tjFlags);

    if (success != 0)
    {
        CORE_ERROR("libjpeg-turbo image conversion failure");
        return 0;
    }
    return ImageJPEG::JpegData{tjJpegBuf};
}
} // namespace core
