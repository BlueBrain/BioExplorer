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

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/OSPRayProperties.h>

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

    _lightData = (ospray::Data*)getParamData(RENDERER_PROPERTY_LIGHTS);
    _lightArray.clear();

    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(((ospray::Light**)_lightData->data)[i]->getIE());

    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];

    _bgMaterial = (AdvancedMaterial*)getParamObject(RENDERER_PROPERTY_BACKGROUND_MATERIAL, nullptr);

    _useHardwareRandomizer =
        getParam(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER.name.c_str(), DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER);

    _exposure = getParam1f(COMMON_PROPERTY_EXPOSURE.name.c_str(), DEFAULT_COMMON_EXPOSURE);
    _randomNumber = getParam1i(OSPRAY_RENDERER_PROPERTY_RANDOM_NUMBER, 0);
    _timestamp = getParam1f(RENDERER_PROPERTY_TIMESTAMP, DEFAULT_RENDERER_TIMESTAMP);

    // Sampling
    _minRayStep = getParam1f("minRayStep", 0.1f);
    _nbRaySteps = getParam1i("nbRaySteps", 8);
    _nbRayRefinementSteps = getParam1i("nbRayRefinementSteps", 8);
    _alphaCorrection = getParam1f(RENDERER_PROPERTY_ALPHA_CORRECTION.name.c_str(), DEFAULT_RENDERER_ALPHA_CORRECTION);

    // Extra
    _cutoff = getParam1f("cutoff", 1.f);

    // Octree
    _userData = getParamData(RENDERER_PROPERTY_USER_DATA);
    _userDataSize = _userData ? _userData->size() : 0;

    // Transfer function
    ospray::TransferFunction* transferFunction =
        (ospray::TransferFunction*)getParamObject(RENDERER_PROPERTY_TRANSFER_FUNCTION, nullptr);
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
