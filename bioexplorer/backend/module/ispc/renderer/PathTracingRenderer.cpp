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

// ::ispc exports
#include "PathTracingRenderer_ispc.h"

using namespace core;

namespace bioexplorer
{
namespace rendering
{
void PathTracingRenderer::commit()
{
    SimulationRenderer::commit();

    _aoWeight = getParam1f(core::engine::ospray::OSPRAY_RENDERER_AMBIENT_OCCLUSION_WEIGHT.name.c_str(), 1.f);
    _aoDistance = getParam1f(core::engine::ospray::OSPRAY_RENDERER_AMBIENT_OCCLUSION_DISTANCE.name.c_str(), 100.f);

    ::ispc::PathTracingRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), spp, _lightPtr,
                                    _lightArray.size(), (_userData ? (float*)_userData->data : nullptr),
                                    _simulationDataSize, _timestamp, _randomNumber, _exposure, _aoWeight, _aoDistance,
                                    _useHardwareRandomizer, _showBackground, _maxRayDepth, _anaglyphEnabled,
                                    (ispc::vec3f&)_anaglyphIpdOffset);
}

PathTracingRenderer::PathTracingRenderer()
{
    ispcEquivalent = ::ispc::PathTracingRenderer_create(this);
}

OSP_REGISTER_RENDERER(PathTracingRenderer, bio_explorer_path_tracing);
OSP_REGISTER_MATERIAL(bio_explorer_path_tracing, core::engine::ospray::AdvancedMaterial, default);
} // namespace rendering
} // namespace bioexplorer
