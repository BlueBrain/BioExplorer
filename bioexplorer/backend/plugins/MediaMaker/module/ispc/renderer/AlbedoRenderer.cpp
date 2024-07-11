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

#include "AlbedoRenderer.h"

#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

#include <ospray/SDK/lights/Light.h>

#include "AlbedoRenderer_ispc.h"

namespace bioexplorer
{
namespace mediamaker
{
namespace rendering
{
void AlbedoRenderer::commit()
{
    SimulationRenderer::commit();

    ::ispc::AlbedoRenderer_set(getIE(), spp, _maxRayDepth, _useHardwareRandomizer,
                               _userData ? (float*)_userData->data : nullptr, _simulationDataSize);
}

AlbedoRenderer::AlbedoRenderer()
{
    ispcEquivalent = ::ispc::AlbedoRenderer_create(this);
}

OSP_REGISTER_RENDERER(AlbedoRenderer, albedo);
OSP_REGISTER_MATERIAL(albedo, core::engine::ospray::AdvancedMaterial, default);
} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer