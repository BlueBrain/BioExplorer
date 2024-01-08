/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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