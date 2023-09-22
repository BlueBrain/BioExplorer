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

#include "GolgiStyleRenderer.h"

#include <science/common/Properties.h>

#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

#include "GolgiStyleRenderer_ispc.h"

namespace bioexplorer
{
namespace rendering
{
void GolgiStyleRenderer::commit()
{
    Renderer::commit();
    _exponent = getParam1f(BIOEXPLORER_RENDERER_PROPERTY_GOLGI_EXPONENT.name.c_str(),
                           BIOEXPLORER_DEFAULT_RENDERER_GOLGI_EXPONENT);
    _inverse =
        getParam(BIOEXPLORER_RENDERER_PROPERTY_GOLGI_INVERSE.name.c_str(), BIOEXPLORER_DEFAULT_RENDERER_GOLGI_INVERSE);

    ::ispc::GolgiStyleRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), spp, _exponent, _inverse);
}

GolgiStyleRenderer::GolgiStyleRenderer()
{
    ispcEquivalent = ::ispc::GolgiStyleRenderer_create(this);
}

OSP_REGISTER_RENDERER(GolgiStyleRenderer, bio_explorer_golgi_style);
OSP_REGISTER_MATERIAL(bio_explorer_golgi_style, core::engine::ospray::AdvancedMaterial, default);
} // namespace rendering
} // namespace bioexplorer
