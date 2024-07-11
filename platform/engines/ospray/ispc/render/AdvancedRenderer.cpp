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

#include "AdvancedRenderer.h"

#include "../geometry/Cones.h"
#include "../geometry/SDFGeometries.h"

#include <platform/core/common/Properties.h>
#include <platform/core/common/geometry/Cone.h>
#include <platform/core/common/geometry/Cylinder.h>
#include <platform/core/common/geometry/SDFGeometry.h>
#include <platform/core/common/geometry/Sphere.h>
#include <platform/engines/ospray/OSPRayProperties.h>

#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/common/Model.h>
#include <ospray/SDK/geometry/Cylinders.h>
#include <ospray/SDK/geometry/Geometry.h>
#include <ospray/SDK/geometry/Spheres.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

#include "AdvancedRenderer_ispc.h"

extern "C"
{
    int AdvancedRenderer_getBytesPerPrimitive(const void* geometry)
    {
        const ::ospray::Geometry* base = static_cast<const ::ospray::Geometry*>(geometry);
        if (dynamic_cast<const ::ospray::Spheres*>(base))
            return sizeof(core::Sphere);
        else if (dynamic_cast<const ::ospray::Cylinders*>(base))
            return sizeof(core::Cylinder);
        else if (dynamic_cast<const ::core::engine::ospray::Cones*>(base))
            return sizeof(::core::Cone);
        else if (dynamic_cast<const ::core::engine::ospray::SDFGeometries*>(base))
            return sizeof(::core::SDFGeometry);
        return 0;
    }
}

namespace core
{
namespace engine
{
namespace ospray
{
void AdvancedRenderer::commit()
{
    SimulationRenderer::commit();

    _fastPreview = getParam(RENDERER_PROPERTY_FAST_PREVIEW.name.c_str(), DEFAULT_RENDERER_FAST_PREVIEW);
    _alphaCorrection = getParam1f(RENDERER_PROPERTY_ALPHA_CORRECTION.name.c_str(), DEFAULT_RENDERER_ALPHA_CORRECTION);
    _shadows = getParam1f(RENDERER_PROPERTY_SHADOW_INTENSITY.name.c_str(), DEFAULT_RENDERER_SHADOW_INTENSITY);
    _softShadows =
        getParam1f(RENDERER_PROPERTY_SOFT_SHADOW_STRENGTH.name.c_str(), DEFAULT_RENDERER_SOFT_SHADOW_STRENGTH);
    _softShadowsSamples = getParam1i(RENDERER_PROPERTY_SHADOW_SAMPLES.name.c_str(), DEFAULT_RENDERER_SHADOW_SAMPLES);
    _giStrength = getParam1f(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_STRENGTH.name.c_str(),
                             DEFAULT_RENDERER_GLOBAL_ILLUMINATION_STRENGTH);
    _giDistance = getParam1f(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_RAY_LENGTH.name.c_str(),
                             DEFAULT_RENDERER_GLOBAL_ILLUMINATION_RAY_LENGTH);
    _giSamples = getParam1i(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_SAMPLES.name.c_str(),
                            DEFAULT_RENDERER_GLOBAL_ILLUMINATION_SAMPLES);
    _matrixFilter = getParam(RENDERER_PROPERTY_MATRIX_FILTER.name.c_str(), DEFAULT_RENDERER_MATRIX_FILTER);
    _volumeSamplingThreshold = getParam1f(OSPRAY_RENDERER_VOLUME_SAMPLING_THRESHOLD.name.c_str(),
                                          OSPRAY_DEFAULT_RENDERER_VOLUME_SAMPLING_THRESHOLD);
    _volumeSpecularExponent = getParam1f(OSPRAY_RENDERER_VOLUME_SPECULAR_EXPONENT.name.c_str(),
                                         OSPRAY_DEFAULT_RENDERER_VOLUME_SPECULAR_EXPONENT);
    _volumeAlphaCorrection = getParam1f(OSPRAY_RENDERER_VOLUME_ALPHA_CORRECTION.name.c_str(),
                                        OSPRAY_DEFAULT_RENDERER_VOLUME_ALPHA_CORRECTION);

    clipPlanes = getParamData(CAMERA_PROPERTY_CLIPPING_PLANES, nullptr);
    const auto clipPlaneData = clipPlanes ? clipPlanes->data : nullptr;
    const uint32 numClipPlanes = clipPlanes ? clipPlanes->numItems : 0;
    ::ispc::AdvancedRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), _shadows, _softShadows,
                                 _softShadowsSamples, _giStrength, _giDistance, _giSamples, _randomNumber, _timestamp,
                                 spp, _lightPtr, _lightArray.size(), _exposure, _epsilonFactor, _fogThickness,
                                 _fogStart, _useHardwareRandomizer, _maxRayDepth, _showBackground, _matrixFilter,
                                 _userData ? (float*)_userData->data : nullptr, _simulationDataSize,
                                 _volumeSamplingThreshold, _volumeSpecularExponent, _volumeAlphaCorrection,
                                 (const ::ispc::vec4f*)clipPlaneData, numClipPlanes, _anaglyphEnabled,
                                 (ispc::vec3f&)_anaglyphIpdOffset, _fastPreview);
}

AdvancedRenderer::AdvancedRenderer()
{
    ispcEquivalent = ::ispc::AdvancedRenderer_create(this);
}

OSP_REGISTER_RENDERER(AdvancedRenderer, advanced);
} // namespace ospray
} // namespace engine
} // namespace core