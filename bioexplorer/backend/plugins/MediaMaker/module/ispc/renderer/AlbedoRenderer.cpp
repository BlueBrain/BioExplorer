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

#include "AlbedoRenderer.h"

#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

// ospray
#include <ospray/SDK/lights/Light.h>

// ispc exports
#include "AlbedoRenderer_ispc.h"

using namespace core;
using namespace ospray;

namespace bioexplorer
{
namespace mediamaker
{
namespace rendering
{
void AlbedoRenderer::commit()
{
    SimulationRenderer::commit();

    ispc::AlbedoRenderer_set(getIE(), spp, _maxBounces, _useHardwareRandomizer,
                             _userData ? (float*)_userData->data : nullptr, _simulationDataSize);
}

AlbedoRenderer::AlbedoRenderer()
{
    ispcEquivalent = ispc::AlbedoRenderer_create(this);
}

OSP_REGISTER_RENDERER(AlbedoRenderer, albedo);
OSP_REGISTER_MATERIAL(albedo, AdvancedMaterial, default);

} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer