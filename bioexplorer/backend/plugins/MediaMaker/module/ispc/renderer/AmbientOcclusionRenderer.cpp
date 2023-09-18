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

#include "AmbientOcclusionRenderer.h"

#include <platform/core/common/Types.h>
#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

// ospray
#include <ospray/SDK/lights/Light.h>

// ispc exports
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
    _samplesPerFrame = getParam1i("samplesPerFrame", 16);
    _aoRayLength = getParam1f("rayLength", 1e6f);
    _maxBounces = getParam1i("maxBounces", 3);
    _useHardwareRandomizer = getParam(RENDERER_PROPERTY_NAME_USE_HARDWARE_RANDOMIZER, 0);

    ispc::AmbientOcclusionRenderer_set(getIE(), spp, _samplesPerFrame, _aoRayLength, _maxBounces,
                                       _useHardwareRandomizer);
}

AmbientOcclusionRenderer::AmbientOcclusionRenderer()
{
    ispcEquivalent = ispc::AmbientOcclusionRenderer_create(this);
}

OSP_REGISTER_RENDERER(AmbientOcclusionRenderer, ambient_occlusion);
OSP_REGISTER_MATERIAL(ambient_occlusion, AdvancedMaterial, default);
} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer