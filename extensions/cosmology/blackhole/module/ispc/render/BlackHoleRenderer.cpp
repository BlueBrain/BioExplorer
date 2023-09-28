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

#include <plugin/common/Properties.h>

#include <platform/core/common/Properties.h>

#include "BlackHoleRenderer_ispc.h"

using namespace ospray;
using namespace core;

namespace spaceexplorer
{
namespace blackhole
{
void BlackHoleRenderer::commit()
{
    AbstractRenderer::commit();

    _grid = getParam(BLACK_HOLE_RENDERER_PROPERTY_DISPLAY_GRID.name.c_str(), BLACK_HOLE_DEFAULT_RENDERER_DISPLAY_GRID);
    _nbDisks = getParam1i(BLACK_HOLE_RENDERER_PROPERTY_NB_DISKS.name.c_str(), BLACK_HOLE_DEFAULT_RENDERER_NB_DISKS);
    _diskRotationSpeed = getParam1f(BLACK_HOLE_RENDERER_PROPERTY_DISK_ROTATION_SPEED.name.c_str(),
                                    BLACK_HOLE_DEFAULT_RENDERER_DISK_ROTATION_SPEED);
    _diskTextureLayers = getParam1i(BLACK_HOLE_RENDERER_PROPERTY_DISK_TEXTURE_LAYERS.name.c_str(),
                                    BLACK_HOLE_DEFAULT_RENDERER_TEXTURE_LAYERS);
    _blackHoleSize = getParam1f(BLACK_HOLE_RENDERER_PROPERTY_SIZE.name.c_str(), BLACK_HOLE_DEFAULT_RENDERER_SIZE);

    ::ispc::BlackHoleRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), _timestamp, spp, _exposure,
                                  _nbDisks, _grid, _diskRotationSpeed, _diskTextureLayers, _blackHoleSize,
                                  _anaglyphEnabled, (ispc::vec3f&)_anaglyphIpdOffset);
}

BlackHoleRenderer::BlackHoleRenderer()
{
    ispcEquivalent = ::ispc::BlackHoleRenderer_create(this);
}

OSP_REGISTER_RENDERER(BlackHoleRenderer, blackhole);
OSP_REGISTER_MATERIAL(blackhole, core::engine::ospray::AdvancedMaterial, default);

} // namespace blackhole
} // namespace spaceexplorer