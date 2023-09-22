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
#include "VoxelRenderer.h"

// ospray
#include <ospray/SDK/common/Data.h>

// ispc exports
#include "VoxelRenderer_ispc.h"

using namespace core;

namespace bioexplorer
{
namespace rendering
{
using namespace ospray;

void VoxelRenderer::commit()
{
    SimulationRenderer::commit();

    _simulationThreshold = getParam1f("simulationThreshold", 0.f);

    ispc::VoxelRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), spp,
                            (_userData ? (float*)_userData->data : nullptr), _simulationDataSize,
                            _alphaCorrection, _simulationThreshold, _exposure);
}

VoxelRenderer::VoxelRenderer()
{
    ispcEquivalent = ispc::VoxelRenderer_create(this);
}

OSP_REGISTER_RENDERER(VoxelRenderer, bio_explorer_voxel);
OSP_REGISTER_MATERIAL(bio_explorer_voxel, AdvancedMaterial, default);

} // namespace rendering
} // namespace bioexplorer
