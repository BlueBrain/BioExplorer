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

#include "ImageManager.h"
#include <OpenImageIO/imageio.h>
#include <filesystem>
#include <iostream>

namespace core
{
OIIO_NAMESPACE_USING

ImageManager::ImageManager()
    : width(0)
    , height(0)
{
}

ImageManager::~ImageManager() {}

bool ImageManager::loadImage(const std::string &filename)
{
    auto in = ImageInput::open(filename);
    if (!in)
    {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return false;
    }
    const ImageSpec &spec = in->spec();
    width = spec.width;
    height = spec.height;
    imageData.resize(width * height * 4);
    in->read_image(TypeDesc::UINT8, &imageData[0]);
    in->close();
    return true;
}

bool ImageManager::saveImage(const std::string &filename)
{
    auto out = ImageOutput::create(filename);
    if (!out)
    {
        std::cerr << "Failed to save image: " << filename << std::endl;
        return false;
    }
    ImageSpec spec(width, height, 4, TypeDesc::UINT8);
    out->open(filename, spec);
    out->write_image(TypeDesc::UINT8, &imageData[0]);
    out->close();
    return true;
}

unsigned int ImageManager::getWidth() const
{
    return width;
}

unsigned int ImageManager::getHeight() const
{
    return height;
}

const std::vector<unsigned char> &ImageManager::getImageData() const
{
    return imageData;
}

Texture2DPtr ImageManager::importTextureFromFile(const std::string &filename, const TextureType type)
{
    auto in = ImageInput::open(filename);
    if (!in)
    {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return {};
    }

    const ImageSpec &spec = in->spec();
    unsigned int width = spec.width;
    unsigned int height = spec.height;
    int channels = spec.nchannels;
    int depth = spec.format.basesize();

    std::vector<unsigned char> pixels(width * height * channels * depth);
    if (!in->read_image(TypeDesc::UINT8, &pixels[0]))
    {
        std::cerr << "Failed to read image data: " << filename << std::endl;
        return {};
    }
    in->close();

    Texture2D::Type textureType = Texture2D::Type::default_;
    const bool isCubeMap = type == TextureType::irradiance || type == TextureType::radiance;
    if (isCubeMap)
    {
        textureType = Texture2D::Type::cubemap;
        width /= 6;
    }
    else if (type == TextureType::normals)
    {
        textureType = Texture2D::Type::normal_roughness;
    }
    else if (type == TextureType::specular)
    {
        textureType = Texture2D::Type::aoe;
    }

    auto texture = std::make_shared<Texture2D>(textureType, filename, channels, depth, width, height);
    if (isCubeMap || type == TextureType::brdf_lut)
    {
        texture->setWrapMode(TextureWrapMode::clamp_to_edge);
    }

    texture->setRawData(pixels.data(), width * height, channels);

    const auto path = std::filesystem::path(filename).parent_path().string();
    const auto basename = path + "/" + std::filesystem::path(filename).stem().string();
    const auto ext = std::filesystem::path(filename).extension().string();

    uint8_t mipLevels = 1;
    while (std::filesystem::exists(basename + std::to_string((int)mipLevels) + ext))
    {
        ++mipLevels;
    }

    texture->setMipLevels(mipLevels);

    for (uint8_t mip = 1; mip < mipLevels; ++mip)
    {
        auto mipFilename = basename + std::to_string((int)mip) + ext;
        auto mipIn = ImageInput::open(mipFilename);
        if (!mipIn)
        {
            std::cerr << "Failed to load mip level image: " << mipFilename << std::endl;
            continue;
        }

        const ImageSpec &mipSpec = mipIn->spec();
        std::vector<unsigned char> mipPixels(mipSpec.width * mipSpec.height * mipSpec.nchannels *
                                             mipSpec.format.basesize());
        if (!mipIn->read_image(TypeDesc::UINT8, &mipPixels[0]))
        {
            std::cerr << "Failed to read mip level image data: " << mipFilename << std::endl;
            mipIn->close();
            continue;
        }
        mipIn->close();

        texture->setRawData(mipPixels.data(), mipSpec.width * mipSpec.height, mipSpec.nchannels, mip);
    }

    return texture;
}

} // namespace core