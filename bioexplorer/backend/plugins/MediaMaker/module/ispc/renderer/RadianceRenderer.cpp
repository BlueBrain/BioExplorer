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

#include "RadianceRenderer.h"

#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

// ospray
#include <ospray/SDK/lights/Light.h>

// ispc exports
#include "RadianceRenderer_ispc.h"

using namespace core;
using namespace ospray;

namespace bioexplorer
{
namespace mediamaker
{
namespace rendering
{
void RadianceRenderer::commit()
{
    Renderer::commit();

    ispc::RadianceRenderer_set(getIE(), spp);
}

RadianceRenderer::RadianceRenderer()
{
    ispcEquivalent = ispc::RadianceRenderer_create(this);
}

OSP_REGISTER_RENDERER(RadianceRenderer, radiance);
OSP_REGISTER_MATERIAL(radiance, AdvancedMaterial, default);

} // namespace rendering
} // namespace mediamaker
} // namespace bioexplorer