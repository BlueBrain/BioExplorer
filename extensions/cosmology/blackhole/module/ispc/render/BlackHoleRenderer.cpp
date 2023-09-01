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

#include "BlackHoleRenderer.h"

// ispc exports
#include "BlackHoleRenderer_ispc.h"

using namespace ospray;

namespace spaceexplorer
{
namespace blackhole
{
void BlackHoleRenderer::commit()
{
    AbstractRenderer::commit();

    _exposure = getParam1f("mainExposure", 1.f);
    _grid = getParam("grid", false);
    _nbDisks = getParam1i("nbDisks", 20);
    _diskRotationSpeed = getParam1f("diskRotationSpeed", 3.f);
    _diskTextureLayers = getParam1i("diskTextureLayers", 12);
    _blackHoleSize = getParam1f("blackHoleSize", 0.3f);

    ispc::BlackHoleRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), _timestamp, spp, _exposure,
                                _nbDisks, _grid, _diskRotationSpeed, _diskTextureLayers, _blackHoleSize);
}

BlackHoleRenderer::BlackHoleRenderer()
{
    ispcEquivalent = ispc::BlackHoleRenderer_create(this);
}

OSP_REGISTER_RENDERER(BlackHoleRenderer, blackhole);
OSP_REGISTER_MATERIAL(blackhole, core::AdvancedMaterial, default);

} // namespace blackhole
} // namespace spaceexplorer