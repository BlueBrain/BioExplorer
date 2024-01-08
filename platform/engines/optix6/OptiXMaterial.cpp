/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "OptiXMaterial.h"

#include "OptiXCommonStructs.h"
#include "OptiXContext.h"
#include "OptiXUtils.h"

#include <platform/core/common/Logs.h>

namespace core
{
namespace engine
{
namespace optix
{
OptiXMaterial::~OptiXMaterial()
{
    for (auto textureSampler : _textureSamplers)
    {
        RT_DESTROY(textureSampler.second);
    }

    RT_DESTROY(_optixMaterial);
}

bool OptiXMaterial::isTextured() const
{
    return !_textureSamplers.empty();
}

void OptiXMaterial::commit()
{
    if (!_optixMaterial)
        _optixMaterial = OptiXContext::get().createMaterial();

    _optixMaterial[CONTEXT_MATERIAL_KA]->setFloat(_emission, _emission, _emission);
    _optixMaterial[CONTEXT_MATERIAL_KD]->setFloat(_diffuseColor.x, _diffuseColor.y, _diffuseColor.z);
    _optixMaterial[CONTEXT_MATERIAL_KS]->setFloat(_specularColor.x, _specularColor.y, _specularColor.z);
    _optixMaterial[CONTEXT_MATERIAL_KR]->setFloat(_reflectionIndex, _reflectionIndex, _reflectionIndex);
    _optixMaterial[CONTEXT_MATERIAL_KO]->setFloat(_opacity, _opacity, _opacity);
    _optixMaterial[CONTEXT_MATERIAL_GLOSSINESS]->setFloat(_glossiness);
    _optixMaterial[CONTEXT_MATERIAL_REFRACTION_INDEX]->setFloat(_refractionIndex);
    _optixMaterial[CONTEXT_MATERIAL_SPECULAR_EXPONENT]->setFloat(_specularExponent);
    _optixMaterial[CONTEXT_MATERIAL_SHADING_MODE]->setUint(_shadingMode);
    _optixMaterial[CONTEXT_MATERIAL_USER_PARAMETER]->setFloat(_userParameter);
    _optixMaterial[CONTEXT_MATERIAL_CAST_USER_DATA]->setUint(_castUserData);
    _optixMaterial[CONTEXT_MATERIAL_CLIPPING_MODE]->setUint(_clippingMode);
    _optixMaterial[CONTEXT_MATERIAL_VALUE_RANGE]->setFloat(_valueRange.x, _valueRange.y);

    for (const auto& textureDescriptor : getTextureDescriptors())
    {
        if (!_textureSamplers.count(textureDescriptor.first))
        {
            const auto textureSampler = OptiXContext::get().createTextureSampler(textureDescriptor.second);
            _textureSamplers.insert(std::make_pair(textureDescriptor.first, textureSampler));
            _optixMaterial[textureTypeToString[static_cast<uint8_t>(textureDescriptor.first)]]->setInt(
                textureSampler->getId());
        }
    }
}
} // namespace optix
} // namespace engine
} // namespace core