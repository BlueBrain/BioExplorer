/*
    Copyright 2015 - 2018 Blue Brain Project / EPFL

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

#include <platform/core/common/Types.h>

#include <turbojpeg.h>

namespace core
{
/**
 * A class which creates images for network communication from a FrameBuffer.
 */
class ImageGenerator
{
public:
    ImageGenerator() = default;
    ~ImageGenerator();

    struct ImageBase64
    {
        std::string data;
    };

    /**
     * Create a base64-encoded image from the given framebuffer in a specified
     * image format and quality.
     *
     * @param frameBuffer the framebuffer to use for getting the pixels
     * @param format FreeImage format string, or JPEG if FreeImage is not
     *               available
     * @param quality image format specific quality number
     * @return base64-encoded image
     * @throw std::runtime_error if image conversion failed or neither FreeImage
     *                           nor TurboJPEG is available
     */
    ImageBase64 createImage(FrameBuffer& frameBuffer, const std::string& format, uint8_t quality);
    ImageBase64 createImage(const std::vector<FrameBufferPtr>& frameBuffers, const std::string& format,
                            uint8_t quality);

    struct ImageJPEG
    {
        struct tjDeleter
        {
            void operator()(uint8_t* ptr) { tjFree(ptr); }
        };
        using JpegData = std::unique_ptr<uint8_t, tjDeleter>;
        JpegData data;
        unsigned long size{0};
    };

    /**
     * Create a JPEG image from the given framebuffer in a specified quality.
     *
     * @param frameBuffer the framebuffer to use for getting the pixels
     * @param quality 1..100 JPEG quality
     * @return JPEG image with a size > 0 if valid, size == 0 on error.
     */
    ImageJPEG createJPEG(FrameBuffer& frameBuffer, uint8_t quality);

private:
    tjhandle _compressor{tjInitCompress()};

    ImageJPEG::JpegData _encodeJpeg(uint32_t width, uint32_t height, const uint8_t* rawData, int32_t pixelFormat,
                                    uint8_t quality, unsigned long& dataSize);
};
} // namespace core
