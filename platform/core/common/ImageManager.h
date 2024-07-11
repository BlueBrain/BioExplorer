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

#pragma once

#include <platform/core/common/Types.h>
#include <platform/core/common/material/Texture2D.h>

namespace core
{

class ImageManager
{
public:
    ImageManager();
    ~ImageManager();

    bool loadImage(const std::string &filename);
    bool saveImage(const std::string &filename);

    unsigned int getWidth() const;
    unsigned int getHeight() const;
    const std::vector<unsigned char> &getImageData() const;
    static Texture2DPtr importTextureFromFile(const std::string &filename, const TextureType type);

private:
    unsigned int width;
    unsigned int height;
    std::vector<unsigned char> imageData;
};

} // namespace core
