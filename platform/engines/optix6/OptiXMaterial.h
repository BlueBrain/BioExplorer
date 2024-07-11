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

#include <map>

#include <platform/core/engineapi/Material.h>

#include <optixu/optixpp_namespace.h>

namespace core
{
namespace engine
{
namespace optix
{
class OptiXMaterial : public Material
{
public:
    OptiXMaterial() = default;
    ~OptiXMaterial();

    void commit() final;
    bool isTextured() const;

    ::optix::Material getOptixMaterial() { return _optixMaterial; }
    auto getTextureSampler(const TextureType type) const { return _textureSamplers.at(type); }
    auto& getTextureSamplers() { return _textureSamplers; }

    void setValueRange(const Vector2f& valueRange) { _valueRange = valueRange; }

private:
    ::optix::Material _optixMaterial{nullptr};
    std::map<TextureType, ::optix::TextureSampler> _textureSamplers;
    Vector2f _valueRange;
};
} // namespace optix
} // namespace engine
} // namespace core