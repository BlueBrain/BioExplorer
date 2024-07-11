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

#include "MetabolismRenderer.h"

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

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

    _lightData = (::ospray::Data*)getParamData(RENDERER_PROPERTY_LIGHTS);
    _lightArray.clear();
    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(((::ospray::Light**)_lightData->data)[i]->getIE());

    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];
    _bgMaterial = (::ospray::Material*)getParamObject(RENDERER_PROPERTY_BACKGROUND_MATERIAL, nullptr);
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
    ::ospray::TransferFunction* transferFunction =
        (::ospray::TransferFunction*)getParamObject(DEFAULT_COMMON_TRANSFER_FUNCTION, nullptr);
    if (transferFunction)
        ::ispc::MetabolismRenderer_setTransferFunction(getIE(), transferFunction->getIE());

    // Renderer
    ::ispc::MetabolismRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), spp, _lightPtr,
                                   _lightArray.size(), (_userData ? (float*)_userData->data : nullptr), _userDataSize,
                                   _nearPlane, _farPlane, _rayStep, _refinementSteps, _exposure, _alphaCorrection,
                                   _colorMapPerRegion, _noiseFrequency, _noiseAmplitude);
}

MetabolismRenderer::MetabolismRenderer()
{
    ispcEquivalent = ::ispc::MetabolismRenderer_create(this);
}

OSP_REGISTER_RENDERER(MetabolismRenderer, metabolism);
OSP_REGISTER_MATERIAL(metabolism, core::engine::ospray::AdvancedMaterial, default);
} // namespace rendering
} // namespace metabolism
