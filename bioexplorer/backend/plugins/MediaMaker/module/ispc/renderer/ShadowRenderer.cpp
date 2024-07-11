/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "ShadowRenderer.h"

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

#include <ospray/SDK/lights/Light.h>

#include "ShadowRenderer_ispc.h"

using namespace core;

namespace bioexplorer
{
namespace mediamaker
{
namespace rendering
{
void ShadowRenderer::commit()
{
    Renderer::commit();

    _lightData = (::ospray::Data*)getParamData(RENDERER_PROPERTY_LIGHTS);
    _lightArray.clear();

    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(((::ospray::Light**)_lightData->data)[i]->getIE());
    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];

    _softness = getParam1f(RENDERER_PROPERTY_SOFT_SHADOW_STRENGTH.name.c_str(), DEFAULT_RENDERER_SOFT_SHADOW_STRENGTH);
    _samplesPerFrame = getParam1i(RENDERER_PROPERTY_SHADOW_SAMPLES.name.c_str(), DEFAULT_RENDERER_SHADOW_SAMPLES);
    _rayLength = getParam1f(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_RAY_LENGTH.name.c_str(),
                            DEFAULT_RENDERER_GLOBAL_ILLUMINATION_RAY_LENGTH);
    ::ispc::ShadowRenderer_set(getIE(), spp, _lightPtr, _lightArray.size(), _samplesPerFrame, _rayLength, _softness);
}

ShadowRenderer::ShadowRenderer()
{
    ispcEquivalent = ::ispc::ShadowRenderer_create(this);
}

OSP_REGISTER_RENDERER(ShadowRenderer, shadow);
OSP_REGISTER_MATERIAL(shadow, core::engine::ospray::AdvancedMaterial, default);
} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer