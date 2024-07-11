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

#include "AmbientOcclusionRenderer.h"

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

// ospray
#include <ospray/SDK/lights/Light.h>

// ::ispc exports
#include "AmbientOcclusionRenderer_ispc.h"

using namespace core;
using namespace ospray;

namespace bioexplorer
{
namespace mediamaker
{
namespace rendering
{
void AmbientOcclusionRenderer::commit()
{
    Renderer::commit();
    _samplesPerFrame = getParam1i(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_SAMPLES.name.c_str(),
                                  DEFAULT_RENDERER_GLOBAL_ILLUMINATION_SAMPLES);
    _aoRayLength = getParam1f(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_RAY_LENGTH.name.c_str(),
                              DEFAULT_RENDERER_GLOBAL_ILLUMINATION_RAY_LENGTH);
    _maxRayDepth = getParam1i(RENDERER_PROPERTY_MAX_RAY_DEPTH.name.c_str(), DEFAULT_RENDERER_MAX_RAY_DEPTH);
    _useHardwareRandomizer = getParam(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER.name.c_str(),
                                      static_cast<int>(DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER));

    ::ispc::AmbientOcclusionRenderer_set(getIE(), spp, _samplesPerFrame, _aoRayLength, _maxRayDepth,
                                         _useHardwareRandomizer);
}

AmbientOcclusionRenderer::AmbientOcclusionRenderer()
{
    ispcEquivalent = ::ispc::AmbientOcclusionRenderer_create(this);
}

OSP_REGISTER_RENDERER(AmbientOcclusionRenderer, ambient_occlusion);
OSP_REGISTER_MATERIAL(ambient_occlusion, core::engine::ospray::AdvancedMaterial, default);
} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer