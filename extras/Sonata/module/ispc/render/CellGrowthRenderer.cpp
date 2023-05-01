/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include "CellGrowthRenderer.h"

// ospray
#include <ospray/SDK/common/Data.h>

// ispc exports
#include "CellGrowthRenderer_ispc.h"

using namespace ospray;

namespace sonataexplorer
{
void CellGrowthRenderer::commit()
{
    SonataExplorerSimulationRenderer::commit();

    _simulationThreshold = getParam1f("simulationThreshold", 0.f);

    _shadows = getParam1f("shadows", 0.f);
    _softShadows = getParam1f("softShadows", 0.f);
    _shadowDistance = getParam1f("shadowDistance", 1e4f);

    _useTransferFunctionColor = getParam("tfColor", false);

    ispc::CellGrowthRenderer_set(
        getIE(), (_secondaryModel ? _secondaryModel->getIE() : nullptr),
        (_bgMaterial ? _bgMaterial->getIE() : nullptr), spp, _lightPtr,
        _lightArray.size(),
        (_simulationData ? (float*)_simulationData->data : nullptr),
        _simulationDataSize, _alphaCorrection, _simulationThreshold, _exposure,
        _fogThickness, _fogStart, _shadows, _softShadows, _shadowDistance,
        _useTransferFunctionColor, _useHardwareRandomizer);
}

CellGrowthRenderer::CellGrowthRenderer()
{
    ispcEquivalent = ispc::CellGrowthRenderer_create(this);
}

OSP_REGISTER_RENDERER(CellGrowthRenderer, circuit_explorer_cell_growth);
} // namespace sonataexplorer
