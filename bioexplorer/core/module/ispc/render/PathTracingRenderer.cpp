/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include "PathTracingRenderer.h"

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>

// ispc exports
#include "PathTracingRenderer_ispc.h"

using namespace ospray;

namespace bioexplorer
{
namespace rendering
{
void PathTracingRenderer::commit()
{
    SimulationRenderer::commit();

    _lightData = (ospray::Data*)getParamData("lights");
    _lightArray.clear();

    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(
                ((ospray::Light**)_lightData->data)[i]->getIE());

    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];

    _bgMaterial = (AdvancedMaterial*)getParamObject("bgMaterial", nullptr);
    _exposure = getParam1f("mainExposure", 1.f);
    _useHardwareRandomizer = getParam("useHardwareRandomizer", 0);
    _showBackground = getParam("showBackground", 0);
    _aoStrength = getParam1f("aoStrength", 1.f);
    _aoDistance = getParam1f("aoDistance", 100.f);
    _randomNumber = rand() % 1000;

    ispc::PathTracingRenderer_set(
        getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), spp, _lightPtr,
        _lightArray.size(),
        (_simulationData ? (float*)_simulationData->data : nullptr),
        _simulationDataSize, _timestamp, _randomNumber, _exposure, _aoStrength,
        _aoDistance, _useHardwareRandomizer, _showBackground);
}

PathTracingRenderer::PathTracingRenderer()
{
    ispcEquivalent = ispc::PathTracingRenderer_create(this);
}

OSP_REGISTER_RENDERER(PathTracingRenderer, bio_explorer_path_tracing);
OSP_REGISTER_MATERIAL(bio_explorer_path_tracing, AdvancedMaterial, default);
} // namespace rendering
} // namespace bioexplorer
