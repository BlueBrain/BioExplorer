/*
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include "MetabolismRenderer.h"

// Platform
#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

// ispc exports
#include "MetabolismRenderer_ispc.h"

using namespace ospray;

namespace metabolism
{
namespace rendering
{
using namespace core;

void MetabolismRenderer::commit()
{
    Renderer::commit();

    _lightData = (ospray::Data*)getParamData(RENDERER_PROPERTY_LIGHTS);
    _lightArray.clear();
    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(((ospray::Light**)_lightData->data)[i]->getIE());

    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];
    _bgMaterial = (ospray::Material*)getParamObject(RENDERER_PROPERTY_BACKGROUND_MATERIAL, nullptr);
    _exposure = getParam1f(COMMON_PROPERTY_EXPOSURE.name.c_str(), DEFAULT_COMMON_EXPOSURE);

    // Sampling
    _nearPlane = getParam1f("nearPlane", 0.f);
    _farPlane = getParam1f("farPlane", 1e6f);
    _rayStep = getParam1f("rayStep", 1.f);
    _alphaCorrection = getParam1f(RENDERER_PROPERTY_ALPHA_CORRECTION.name.c_str(), DEFAULT_RENDERER_ALPHA_CORRECTION);
    _refinementSteps = getParam1i("refinementSteps", 64);
    _colorMapPerRegion = getParam("colorMapPerRegion", 0);
    _noiseFrequency = getParam1f("noiseFrequency", 1.f);
    _noiseAmplitude = getParam1f("noiseAmplitude", 1.f);

    _userData = getParamData(RENDERER_PROPERTY_USER_DATA);
    _userDataSize = _userData ? _userData->size() : 0;

    // Transfer function
    ospray::TransferFunction* transferFunction =
        (ospray::TransferFunction*)getParamObject(RENDERER_PROPERTY_TRANSFER_FUNCTION, nullptr);
    if (transferFunction)
        ispc::MetabolismRenderer_setTransferFunction(getIE(), transferFunction->getIE());

    // Renderer
    ispc::MetabolismRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), spp, _lightPtr,
                                 _lightArray.size(), (_userData ? (float*)_userData->data : nullptr), _userDataSize,
                                 _nearPlane, _farPlane, _rayStep, _refinementSteps, _exposure, _alphaCorrection,
                                 _colorMapPerRegion, _noiseFrequency, _noiseAmplitude);
}

MetabolismRenderer::MetabolismRenderer()
{
    ispcEquivalent = ispc::MetabolismRenderer_create(this);
}

OSP_REGISTER_RENDERER(MetabolismRenderer, metabolism);
OSP_REGISTER_MATERIAL(metabolism, AdvancedMaterial, default);

} // namespace rendering
} // namespace metabolism
