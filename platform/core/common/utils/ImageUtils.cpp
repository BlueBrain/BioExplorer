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

#include "ImageUtils.h"

#include <platform/core/common/Logs.h>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <OpenImageIO/imageio.h>

#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/evp.h>

OIIO_NAMESPACE_USING

namespace core
{
bool swapRedBlue32(ImageBuf& image)
{
    if (image.spec().nchannels < 3)
        CORE_THROW("Image must have at least 3 channels to swap red and blue")

    int width = image.spec().width;
    int height = image.spec().height;
    int channels = image.spec().nchannels;

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            float pixel[channels];
            image.getpixel(x, y, pixel);
            std::swap(pixel[0], pixel[2]); // Swap red and blue channels
            image.setpixel(x, y, pixel);
        }

    return true;
}
std::string base64_encode(const unsigned char* data, size_t len)
{
    BIO* bmem = nullptr;
    BIO* b64 = BIO_new(BIO_f_base64());
    BUF_MEM* bptr = nullptr;

    b64 = BIO_push(b64, BIO_new(BIO_s_mem()));
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    BIO_write(b64, data, len);
    BIO_flush(b64);
    BIO_get_mem_ptr(b64, &bptr);

    std::string base64_str(bptr->data, bptr->length);
    BIO_free_all(b64);

    return base64_str;
}

std::string getBase64Image(const ImageBuf& imageBuf, const std::string& format, const int quality)
{
    ImageBuf rotatedBuf;
    ImageBufAlgo::flip(rotatedBuf, imageBuf);
    swapRedBlue32(rotatedBuf);

    // Create a temporary file
    std::string temp_filename = Filesystem::unique_path() + "." + format;

    auto out = ImageOutput::create(temp_filename);
    if (!out)
        CORE_THROW("Failed to create image output");

    ImageSpec spec = rotatedBuf.spec();
    spec.attribute("oiio:CompressionQuality", quality);

    // Open the output for writing to the temporary file
    if (!out->open(temp_filename, spec))
        CORE_THROW("Failed to open image output for temporary file");

    // Write the image to the temporary file
    if (!out->write_image(TypeDesc::UINT8, rotatedBuf.localpixels()))
        CORE_THROW("Failed to write image to temporary file");

    out->close();
    // delete out;

    // Read the file back into memory
    std::ifstream file(temp_filename, std::ios::binary);
    if (!file)
        CORE_THROW("Failed to open temporary file for reading");

    std::vector<unsigned char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Remove the temporary file
    Filesystem::remove(temp_filename);

    // Encode the memory data to base64
    return base64_encode(buffer.data(), buffer.size());
}

ImageBuf mergeImages(const std::vector<ImageBuf>& images)
{
    if (images.empty())
    {
        throw std::runtime_error("No images to merge.");
    }

    int totalWidth = 0;
    int height = images[0].spec().height;
    int channels = images[0].spec().nchannels;

    for (const auto& image : images)
    {
        if (image.spec().height != height || image.spec().nchannels != channels)
            CORE_THROW("All images must have the same height and number of channels.");
        totalWidth += image.spec().width;
    }

    ImageSpec mergedSpec(totalWidth, height, channels, TypeDesc::UINT8);
    ImageBuf mergedImage(mergedSpec);
    int offsetX = 0;

    for (const auto& image : images)
    {
        int imageWidth = image.spec().width;
        ROI roi(offsetX, offsetX + imageWidth, 0, height);
        ImageBufAlgo::paste(mergedImage, offsetX, 0, 0, 0, image, roi);
        offsetX += imageWidth;
    }

    return mergedImage;
}
} // namespace core
