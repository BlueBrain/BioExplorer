/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
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
