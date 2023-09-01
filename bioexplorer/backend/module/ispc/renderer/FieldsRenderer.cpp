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

#include "FieldsRenderer.h"

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

// ispc exports
#include "FieldsRenderer_ispc.h"

using namespace core;

namespace bioexplorer
{
namespace rendering
{
void FieldsRenderer::commit()
{
    Renderer::commit();

    _lightData = (ospray::Data*)getParamData("lights");
    _lightArray.clear();

    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(((ospray::Light**)_lightData->data)[i]->getIE());

    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];

    _bgMaterial = (AdvancedMaterial*)getParamObject("bgMaterial", nullptr);

    _useHardwareRandomizer = getParam("useHardwareRandomizer", 0);

    _exposure = getParam1f("mainExposure", 1.f);
    _randomNumber = getParam1i("randomNumber", 0);
    _timestamp = getParam1f("timestamp", 0.f);

    // Sampling
    _minRayStep = getParam1f("minRayStep", 0.1f);
    _nbRaySteps = getParam1i("nbRaySteps", 8);
    _nbRayRefinementSteps = getParam1i("nbRayRefinementSteps", 8);
    _alphaCorrection = getParam1f("alphaCorrection", 1.0f);

    // Extra
    _cutoff = getParam1f("cutoff", 1.f);

    // Octree
    _userData = getParamData("simulationData");
    _userDataSize = _userData ? _userData->size() : 0;

    // Transfer function
    ospray::TransferFunction* transferFunction = (ospray::TransferFunction*)getParamObject("transferFunction", nullptr);
    if (transferFunction)
        ispc::FieldsRenderer_setTransferFunction(getIE(), transferFunction->getIE());

    // Renderer
    ispc::FieldsRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr),
                             (_userData ? (float*)_userData->data : nullptr), _userDataSize, _randomNumber, _timestamp,
                             spp, _lightPtr, _lightArray.size(), _minRayStep, _nbRaySteps, _nbRayRefinementSteps,
                             _exposure, _useHardwareRandomizer, _cutoff, _alphaCorrection);
}

FieldsRenderer::FieldsRenderer()
{
    ispcEquivalent = ispc::FieldsRenderer_create(this);
}

OSP_REGISTER_RENDERER(FieldsRenderer, bio_explorer_fields);
OSP_REGISTER_MATERIAL(bio_explorer_fields, AdvancedMaterial, default);
} // namespace rendering
} // namespace bioexplorer
