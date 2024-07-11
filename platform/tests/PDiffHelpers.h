#pragma once

/*
    Copyright 2018 - 0211 Blue Brain Project / EPFL

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

#include <Defines.h>

#include <platform/core/Core.h>
#include <platform/core/common/utils/ImageUtils.h>
#include <platform/core/engineapi/FrameBuffer.h>

#include <deps/perceptualdiff/metric.h>
#include <deps/perceptualdiff/rgba_image.h>

#include <tests/paths.h>

#include <iostream>

#ifdef USE_NETWORKING
#include <ImageGenerator.h>
#include <fstream>
#include <platform/core/common/utils/base64/base64.h>
#endif

inline std::unique_ptr<pdiff::RGBAImage> createPDiffRGBAImage(FIBITMAP* image)
{
    const auto w = FreeImage_GetWidth(image);
    const auto h = FreeImage_GetHeight(image);

    auto result = std::make_unique<pdiff::RGBAImage>(w, h, "");
    // Copy the image over to our internal format, FreeImage has the scanlines
    // bottom to top though.
    auto dest = result->get_data();
    for (unsigned int y = 0; y < h; y++, dest += w)
    {
        const auto scanline = reinterpret_cast<const unsigned int*>(FreeImage_GetScanLine(image, h - y - 1));
        memcpy(dest, scanline, sizeof(dest[0]) * w);
    }

    return result;
}

inline std::unique_ptr<pdiff::RGBAImage> createPDiffRGBAImage(core::FrameBuffer& fb)
{
    return createPDiffRGBAImage(FreeImage_ConvertTo32Bits(fb.getImage().get()));
}

inline std::unique_ptr<pdiff::RGBAImage> clonePDiffRGBAImage(const pdiff::RGBAImage& image)
{
    auto result = std::make_unique<pdiff::RGBAImage>(image.get_width(), image.get_height(), "");
    const auto dataSize = image.get_width() * image.get_height() * 4;
    memcpy(result->get_data(), image.get_data(), dataSize);
    return result;
}

inline bool _compareImage(const pdiff::RGBAImage& image, const std::string& filename, pdiff::RGBAImage& imageDiff,
                          const pdiff::PerceptualDiffParameters& parameters = pdiff::PerceptualDiffParameters())
{
    const auto fullPath = std::string(BRAYNS_TESTDATA_IMAGES_PATH) + filename;
    const auto referenceImage{pdiff::read_from_file(fullPath)};
    std::string errorOutput;
    bool success =
        pdiff::yee_compare(*referenceImage, image, parameters, nullptr, nullptr, &errorOutput, &imageDiff, nullptr);
    if (!success)
        std::cerr << "Pdiff failure reason: " << errorOutput;
    return success;
}

inline bool compareTestImage(const std::string& filename, core::FrameBuffer& fb,
                             const pdiff::PerceptualDiffParameters& parameters = pdiff::PerceptualDiffParameters())
{
    static bool saveTestImages = getenv("BRAYNS_SAVE_TEST_IMAGES");
    static bool saveDiffImages = getenv("BRAYNS_SAVE_DIFF_IMAGES");
    if (saveTestImages)
        createPDiffRGBAImage(fb)->write_to_file(filename);

    auto testImage = createPDiffRGBAImage(fb);
    auto imageDiff = clonePDiffRGBAImage(*testImage);

    bool success = _compareImage(*testImage, filename, *imageDiff, parameters);

    if (!success && saveDiffImages)
    {
        const auto filenameDiff = ("diff_" + filename);
        imageDiff->write_to_file(filenameDiff);
    }

    return success;
}

#ifdef USE_NETWORKING
inline bool compareBase64TestImage(const core::ImageGenerator::ImageBase64& image, const std::string& filename)
{
    auto decodedImage = base64_decode(image.data);

    static bool saveImages = getenv("BRAYNS_SAVE_TEST_IMAGES");
    if (saveImages)
    {
        std::fstream file(filename, std::ios::out);
        file << decodedImage;
    }

    auto freeImageMem = FreeImage_OpenMemory((BYTE*)decodedImage.data(), decodedImage.length());
    const auto fif = FreeImage_GetFileTypeFromMemory(freeImageMem, 0);
    auto decodedFreeImage = FreeImage_LoadFromMemory(fif, freeImageMem, 0);

    const auto testImage{createPDiffRGBAImage(decodedFreeImage)};
    auto imageDiff = clonePDiffRGBAImage(*testImage);

    auto result = _compareImage(*testImage, filename, *imageDiff);

    FreeImage_Unload(decodedFreeImage);
    FreeImage_CloseMemory(freeImageMem);

    return result;
}
#endif
