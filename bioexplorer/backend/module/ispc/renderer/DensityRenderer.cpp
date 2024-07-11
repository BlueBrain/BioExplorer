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

#include "DensityRenderer.h"

#include <platform/core/common/Properties.h>
#include <science/common/Properties.h>

#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

#include "DensityRenderer_ispc.h"

using namespace core;

namespace bioexplorer
{
namespace rendering
{
void DensityRenderer::commit()
{
    AbstractRenderer::commit();
    _exposure = getParam1f(COMMON_PROPERTY_EXPOSURE.name.c_str(), DEFAULT_COMMON_EXPOSURE);
    _farPlane =
        getParam1f(BIOEXPLORER_RENDERER_PROPERTY_FAR_PLANE.name.c_str(), BIOEXPLORER_DEFAULT_RENDERER_FAR_PLANE);
    _rayStep = getParam1f(BIOEXPLORER_RENDERER_PROPERTY_RAY_STEP.name.c_str(), BIOEXPLORER_DEFAULT_RENDERER_RAY_STEP);
    _samplesPerFrame = getParam1i(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_SAMPLES.name.c_str(),
                                  DEFAULT_RENDERER_GLOBAL_ILLUMINATION_RAY_LENGTH);
    _searchLength = getParam1f(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_STRENGTH.name.c_str(),
                               DEFAULT_RENDERER_GLOBAL_ILLUMINATION_RAY_LENGTH);
    _alphaCorrection = getParam1f(RENDERER_PROPERTY_ALPHA_CORRECTION.name.c_str(), DEFAULT_RENDERER_ALPHA_CORRECTION);

    // Transfer function
    ::ospray::TransferFunction* transferFunction =
        (::ospray::TransferFunction*)getParamObject(DEFAULT_COMMON_TRANSFER_FUNCTION, nullptr);
    if (transferFunction)
        ::ispc::DensityRenderer_setTransferFunction(getIE(), transferFunction->getIE());

    ::ispc::DensityRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), _timestamp, spp, _farPlane,
                                _searchLength, _rayStep, _samplesPerFrame, _exposure, _alphaCorrection,
                                _anaglyphEnabled, (ispc::vec3f&)_anaglyphIpdOffset);
}

DensityRenderer::DensityRenderer()
{
    ispcEquivalent = ::ispc::DensityRenderer_create(this);
}

OSP_REGISTER_RENDERER(DensityRenderer, bio_explorer_density);
OSP_REGISTER_MATERIAL(bio_explorer_density, core::engine::ospray::AdvancedMaterial, default);
} // namespace rendering
} // namespace bioexplorer
