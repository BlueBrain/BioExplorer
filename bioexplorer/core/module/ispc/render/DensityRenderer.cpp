/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include "DensityRenderer.h"

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

// ispc exports
#include "DensityRenderer_ispc.h"

namespace bioexplorer
{
namespace rendering
{
void DensityRenderer::commit()
{
    Renderer::commit();

    _bgMaterial = (AdvancedMaterial*)getParamObject("bgMaterial", nullptr);

    _exposure = getParam1f("exposure", 1.f);
    _timestamp = getParam1f("timestamp", 0.f);

    // Sampling
    _farPlane = getParam1f("farPlane", 1e6f);
    _rayStep = getParam1f("rayStep", 1.f);
    _sampleCount = getParam1i("sampleCount", 8);
    _searchLength = getParam1f("searchLength", 100.f);
    _alphaCorrection = getParam1f("alphaCorrection", 1.0f);

    // Transfer function
    ospray::TransferFunction* transferFunction =
        (ospray::TransferFunction*)getParamObject("transferFunction", nullptr);
    if (transferFunction)
        ispc::DensityRenderer_setTransferFunction(getIE(),
                                                  transferFunction->getIE());

    // Renderer
    ispc::DensityRenderer_set(getIE(),
                              (_bgMaterial ? _bgMaterial->getIE() : nullptr),
                              _timestamp, spp, _farPlane, _searchLength,
                              _rayStep, _sampleCount, _exposure,
                              _alphaCorrection);
}

DensityRenderer::DensityRenderer()
{
    ispcEquivalent = ispc::DensityRenderer_create(this);
}

OSP_REGISTER_RENDERER(DensityRenderer, bio_explorer_density);
} // namespace rendering
} // namespace bioexplorer
