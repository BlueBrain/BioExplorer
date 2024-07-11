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

#include "AbstractRenderer.h"

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/OSPRayProperties.h>

#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>

namespace core
{
namespace engine
{
namespace ospray
{
void AbstractRenderer::commit()
{
    Renderer::commit();

    _lightData = (::ospray::Data*)getParamData(RENDERER_PROPERTY_LIGHTS);
    _lightArray.clear();
    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(((::ospray::Light**)_lightData->data)[i]->getIE());
    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];

    _timestamp = getParam1f(RENDERER_PROPERTY_TIMESTAMP.name.c_str(), DEFAULT_RENDERER_TIMESTAMP);
    _bgMaterial = (AdvancedMaterial*)getParamObject(RENDERER_PROPERTY_BACKGROUND_MATERIAL, nullptr);
    _useHardwareRandomizer = getParam(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER.name.c_str(),
                                      static_cast<int>(DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER));
    _anaglyphEnabled = getParam(OSPRAY_RENDERER_PROPERTY_ANAGLYPH_ENABLED, DEFAULT_RENDERER_ANAGLYPH_ENABLED);
    _anaglyphIpdOffset = getParam3f(OSPRAY_RENDERER_PROPERTY_ANAGLYPH_IPD_OFFSET, ::ospray::vec3f());
}
} // namespace ospray
} // namespace engine
} // namespace core