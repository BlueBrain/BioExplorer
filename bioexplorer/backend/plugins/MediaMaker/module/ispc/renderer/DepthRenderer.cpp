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

#include "DepthRenderer.h"

#include <plugin/common/Properties.h>

// ospray
#include <ospray/SDK/lights/Light.h>

// ispc exports
#include "DepthRenderer_ispc.h"

using namespace ospray;
using namespace core;

namespace bioexplorer
{
namespace mediamaker
{
namespace rendering
{
void DepthRenderer::commit()
{
    Renderer::commit();

    _infinity = getParam1f(MEDIA_MAKER_RENDERER_PROPERTY_DEPTH_INFINITY.name.c_str(),
                           DEFAULT_MEDIA_MAKER_RENDERER_DEPTH_INFINITY);
    ispc::DepthRenderer_set(getIE(), spp, _infinity);
}

DepthRenderer::DepthRenderer()
{
    ispcEquivalent = ispc::DepthRenderer_create(this);
}

OSP_REGISTER_RENDERER(DepthRenderer, depth);
} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer