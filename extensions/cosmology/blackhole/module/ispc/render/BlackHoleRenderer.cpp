/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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