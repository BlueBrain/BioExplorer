/*
 * Copyright 2020-2024 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <common/Properties.h>
#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

#include <ospray/SDK/common/Data.h>

#include "CellGrowthRenderer_ispc.h"

using namespace core;

namespace sonataexplorer
{
void CellGrowthRenderer::commit()
{
    SimulationRenderer::commit();

    _simulationThreshold = getParam1f(SONATA_RENDERER_PROPERTY_CELL_GROWTH_SIMULATION_THRESHOLD.name.c_str(),
                                      SONATA_DEFAULT_RENDERER_CELL_GROWTH_USE_TRANSFER_FUNCTION_COLOR);
    _shadows = getParam1f(RENDERER_PROPERTY_SHADOW_INTENSITY.name.c_str(), DEFAULT_RENDERER_SHADOW_INTENSITY);
    _softShadows =
        getParam1f(RENDERER_PROPERTY_SOFT_SHADOW_STRENGTH.name.c_str(), DEFAULT_RENDERER_SOFT_SHADOW_STRENGTH);
    _shadowDistance = getParam1f(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_RAY_LENGTH.name.c_str(),
                                 DEFAULT_RENDERER_GLOBAL_ILLUMINATION_RAY_LENGTH);
    _useTransferFunctionColor = getParam(SONATA_RENDERER_PROPERTY_USE_TRANSFER_FUNCTION_COLOR.name.c_str(),
                                         SONATA_DEFAULT_RENDERER_CELL_GROWTH_USE_TRANSFER_FUNCTION_COLOR);
    ::ispc::CellGrowthRenderer_set(getIE(), (_secondaryModel ? _secondaryModel->getIE() : nullptr),
                                   (_bgMaterial ? _bgMaterial->getIE() : nullptr), spp, _lightPtr, _lightArray.size(),
                                   (_userData ? (float*)_userData->data : nullptr), _simulationDataSize,
                                   _alphaCorrection, _simulationThreshold, _exposure, _fogThickness, _fogStart,
                                   _shadows, _softShadows, _shadowDistance, _useTransferFunctionColor,
                                   _useHardwareRandomizer, _anaglyphEnabled, (ispc::vec3f&)_anaglyphIpdOffset);
}

CellGrowthRenderer::CellGrowthRenderer()
{
    ispcEquivalent = ::ispc::CellGrowthRenderer_create(this);
}

OSP_REGISTER_RENDERER(CellGrowthRenderer, cell_growth);
OSP_REGISTER_MATERIAL(cell_growth, core::engine::ospray::AdvancedMaterial, default);
} // namespace sonataexplorer
