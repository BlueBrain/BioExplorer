/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
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

#include "OSPRayMaterial.h"
#include "OSPRayProperties.h"
#include "OSPRayUtils.h"

#include <platform/core/common/Logs.h>
#include <platform/core/common/Properties.h>

#include <ospray/SDK/common/OSPCommon.h>

#include <cassert>

namespace core
{
namespace engine
{
namespace ospray
{
struct TextureTypeMaterialAttribute
{
    TextureType type;
    std::string attribute;
};

static TextureTypeMaterialAttribute textureTypeMaterialAttribute[8] = {
    {TextureType::diffuse, MATERIAL_PROPERTY_MAP_DIFFUSE_COLOR},
    {TextureType::normals, MATERIAL_PROPERTY_MAP_BUMP},
    {TextureType::bump, MATERIAL_PROPERTY_MAP_BUMP},
    {TextureType::specular, MATERIAL_PROPERTY_MAP_SPECULAR_INDEX},
    {TextureType::emissive, MATERIAL_PROPERTY_MAP_EMISSION},
    {TextureType::opacity, MATERIAL_PROPERTY_OPACITY},
    {TextureType::reflection, MATERIAL_PROPERTY_MAP_REFLECTION},
    {TextureType::refraction, MATERIAL_PROPERTY_MAP_REFRACTION}};

OSPRayMaterial::~OSPRayMaterial()
{
    ospRelease(_ospMaterial);
}

void OSPRayMaterial::commit()
{
    // Do nothing until this material is instanced for a specific renderer
    if (!_ospMaterial || !isModified())
        return;

    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_DIFFUSE_COLOR, Vector3f(_diffuseColor));
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_SPECULAR_COLOR, Vector3f(_specularColor));
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_SPECULAR_INDEX, static_cast<float>(_specularExponent));
#if 0
    // For some unknown reason, this simply does not work!?!
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_OPACITY, static_cast<float>(_opacity));
#else
    osphelper::set(_ospMaterial, "opacity", static_cast<float>(_opacity));
#endif
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_REFRACTION, static_cast<float>(_refractionIndex));
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_REFLECTION, static_cast<float>(_reflectionIndex));
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_EMISSION, static_cast<float>(_emission));
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_GLOSSINESS, static_cast<float>(_glossiness));
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_SHADING_MODE, static_cast<MaterialShadingMode>(_shadingMode));
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_USER_PARAMETER, static_cast<float>(_userParameter));
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_CAST_USER_DATA, static_cast<int32_t>(_castUserData));
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_CLIPPING_MODE, static_cast<MaterialClippingMode>(_clippingMode));
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_CHAMELEON_MODE, static_cast<MaterialChameleonMode>(_chameleonMode));
    osphelper::set(_ospMaterial, MATERIAL_PROPERTY_NODE_ID, static_cast<int32_t>(_nodeId));

    // Properties
    toOSPRayProperties(*this, _ospMaterial);

    // Textures
    for (const auto& textureType : textureTypeMaterialAttribute)
        ospSetObject(_ospMaterial, textureType.attribute.c_str(), nullptr);

    for (const auto& textureDescriptor : _textureDescriptors)
    {
        const auto texType = textureDescriptor.first;
        auto texture = getTexture(texType);
        if (texture)
        {
            auto ospTexture = _createOSPTexture2D(texture);
            const auto str = textureTypeMaterialAttribute[int(texType)].attribute.c_str();
            ospSetObject(_ospMaterial, str, ospTexture);
            ospRelease(ospTexture);
        }
    }

    ospCommit(_ospMaterial);
    resetModified();
}

void OSPRayMaterial::commit(const std::string& renderer)
{
    if (!isModified() && renderer == _renderer)
        return;

    ospRelease(_ospMaterial);
    _ospMaterial = ospNewMaterial2(renderer.c_str(), DEFAULT);
    if (!_ospMaterial)
        throw std::runtime_error("Could not create material for renderer '" + renderer + "'");
    _renderer = renderer;
    markModified(false); // Ensure commit recreates the ISPC object
    commit();
}

OSPTexture OSPRayMaterial::_createOSPTexture2D(Texture2DPtr texture)
{
    OSPTextureFormat type = OSP_TEXTURE_R8; // smallest valid type as default
    if (texture->depth == 1)
    {
        if (texture->channels == 1)
            type = OSP_TEXTURE_R8;
        if (texture->channels == 3)
            type = OSP_TEXTURE_RGB8;
        if (texture->channels == 4)
            type = OSP_TEXTURE_RGBA8;
    }
    else if (texture->depth == 4)
    {
        if (texture->channels == 1)
            type = OSP_TEXTURE_R32F;
        if (texture->channels == 3)
            type = OSP_TEXTURE_RGB32F;
        if (texture->channels == 4)
            type = OSP_TEXTURE_RGBA32F;
    }

    CORE_DEBUG("Creating OSPRay texture from " << texture->filename << ": " << texture->width << "x" << texture->height
                                               << "x" << (int)type);

    OSPTexture ospTexture = ospNewTexture(OSPRAY_MATERIAL_TEXTURE_2D);

    const Vector2i size{int(texture->width), int(texture->height)};

    osphelper::set(ospTexture, OSPRAY_MATERIAL_PROPERTY_TEXTURE_TYPE, static_cast<int>(type));
    osphelper::set(ospTexture, OSPRAY_MATERIAL_PROPERTY_TEXTURE_SIZE, size);
    auto textureData =
        ospNewData(texture->getSizeInBytes(), OSP_RAW, texture->getRawData<unsigned char>(), OSP_DATA_SHARED_BUFFER);
    ospSetObject(ospTexture, OSPRAY_MATERIAL_PROPERTY_TEXTURE_DATA, textureData);
    ospRelease(textureData);
    ospCommit(ospTexture);

    return ospTexture;
}
} // namespace ospray
} // namespace engine
} // namespace core