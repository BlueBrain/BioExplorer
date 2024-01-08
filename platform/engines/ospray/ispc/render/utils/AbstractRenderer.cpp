/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Based on OSPRay implementation
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