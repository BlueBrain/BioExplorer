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

#include "Material.h"

#include <platform/core/common/ImageManager.h>
#include <platform/core/common/Logs.h>
#include <platform/core/common/Properties.h>

namespace core
{
Material::Material(const PropertyMap& properties)
{
    setCurrentType(DEFAULT);
    _properties.at(_currentType).merge(properties);
}

Texture2DPtr Material::getTexture(const TextureType type) const
{
    const auto it = _textureDescriptors.find(type);
    if (it == _textureDescriptors.end())
        throw std::runtime_error("Failed to get texture with type " + std::to_string(static_cast<int>(type)));
    return it->second;
}

void Material::clearTextures()
{
    _textureDescriptors.clear();
    markModified();
}

bool Material::_loadTexture(const std::string& fileName, const TextureType type)
{
    if (_textures.find(fileName) != _textures.end())
        return true;

    auto texture = ImageManager::importTextureFromFile(fileName, type);
    if (!texture)
        return false;

    _textures[fileName] = texture;
    CORE_DEBUG(fileName << ": " << texture->width << "x" << texture->height << "x" << (int)texture->channels << "x"
                        << (int)texture->depth << " added to the texture cache");
    return true;
}

void Material::setTexture(const std::string& fileName, const TextureType type)
{
    auto i = _textureDescriptors.find(type);
    if (i != _textureDescriptors.end() && i->second->filename == fileName)
        return;

    if (_textures.find(fileName) == _textures.end())
        if (!_loadTexture(fileName, type))
            throw std::runtime_error("Failed to load texture from " + fileName);
    _textureDescriptors[type] = _textures[fileName];
    markModified();
}

void Material::removeTexture(const TextureType type)
{
    auto i = _textureDescriptors.find(type);
    if (i == _textureDescriptors.end())
        return;

    _textureDescriptors.erase(i);
    markModified();
}
} // namespace core
