/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "PathTracingRenderer.h"

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/OSPRayProperties.h>

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>

// ispc exports
#include "PathTracingRenderer_ispc.h"

using namespace ospray;
using namespace core;

namespace bioexplorer
{
namespace rendering
{
void PathTracingRenderer::commit()
{
    SimulationRenderer::commit();

    _lightData = (ospray::Data*)getParamData(RENDERER_PROPERTY_LIGHTS);
    _lightArray.clear();

    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(((ospray::Light**)_lightData->data)[i]->getIE());

    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];

    _bgMaterial = (AdvancedMaterial*)getParamObject(RENDERER_PROPERTY_BACKGROUND_MATERIAL, nullptr);
    _exposure = getParam1f(COMMON_PROPERTY_EXPOSURE.name.c_str(), DEFAULT_COMMON_EXPOSURE);
    _useHardwareRandomizer = getParam(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER.name.c_str(),
                                      static_cast<int>(DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER));
    _showBackground = getParam(RENDERER_PROPERTY_SHOW_BACKGROUND.name.c_str(), DEFAULT_RENDERER_SHOW_BACKGROUND);
    _aoWeight = getParam1f(OSPRAY_RENDERER_AMBIENT_OCCLUSION_WEIGHT.name.c_str(), 1.f);
    _aoDistance = getParam1f(OSPRAY_RENDERER_AMBIENT_OCCLUSION_DISTANCE.name.c_str(), 100.f);
    _randomNumber = rand() % 1000;

    ispc::PathTracingRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), spp, _lightPtr,
                                  _lightArray.size(), (_userData ? (float*)_userData->data : nullptr),
                                  _simulationDataSize, _timestamp, _randomNumber, _exposure, _aoWeight, _aoDistance,
                                  _useHardwareRandomizer, _showBackground);
}

PathTracingRenderer::PathTracingRenderer()
{
    ispcEquivalent = ispc::PathTracingRenderer_create(this);
}

OSP_REGISTER_RENDERER(PathTracingRenderer, bio_explorer_path_tracing);
OSP_REGISTER_MATERIAL(bio_explorer_path_tracing, AdvancedMaterial, default);
} // namespace rendering
} // namespace bioexplorer
