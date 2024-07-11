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
