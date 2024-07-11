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
        RT_DESTROY(textureSampler.second);

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

    const auto& textureDescriptors = getTextureDescriptors();
    for (const auto& textureDescriptor : textureDescriptors)
        if (!_textureSamplers.count(textureDescriptor.first))
        {
            const auto textureSampler = OptiXContext::get().createTextureSampler(textureDescriptor.second);
            _textureSamplers.insert(std::make_pair(textureDescriptor.first, textureSampler));
            _optixMaterial[textureTypeToString[static_cast<uint8_t>(textureDescriptor.first)]]->setInt(
                textureSampler->getId());
        }
}
} // namespace optix
} // namespace engine
} // namespace core