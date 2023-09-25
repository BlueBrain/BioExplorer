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

    _alphaCorrection = getParam1f(RENDERER_PROPERTY_ALPHA_CORRECTION.name.c_str(), DEFAULT_RENDERER_ALPHA_CORRECTION);
    _fogStart = getParam1f(RENDERER_PROPERTY_FOG_START.name.c_str(), DEFAULT_RENDERER_FOG_START);
    _fogThickness = getParam1f(RENDERER_PROPERTY_FOG_THICKNESS.name.c_str(), DEFAULT_RENDERER_FOG_THICKNESS);
    _exposure = getParam1f(COMMON_PROPERTY_EXPOSURE.name.c_str(), DEFAULT_COMMON_EXPOSURE);
    _epsilonFactor = getParam1f(RENDERER_PROPERTY_EPSILON_MULTIPLIER.name.c_str(), DEFAULT_RENDERER_EPSILON_MULTIPLIER);
    _maxBounces = getParam1i(RENDERER_PROPERTY_MAX_RAY_DEPTH.name.c_str(), DEFAULT_RENDERER_MAX_RAY_DEPTH);
    _randomNumber = rand() % 1000;
    _useHardwareRandomizer = getParam(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER.name.c_str(),
                                      static_cast<int>(DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER));
    _showBackground = getParam(RENDERER_PROPERTY_SHOW_BACKGROUND.name.c_str(), DEFAULT_RENDERER_SHOW_BACKGROUND);
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
                                 _fogStart, _useHardwareRandomizer, _maxBounces, _showBackground, _matrixFilter,
                                 _userData ? (float*)_userData->data : nullptr, _simulationDataSize,
                                 _volumeSamplingThreshold, _volumeSpecularExponent, _volumeAlphaCorrection,
                                 (const ::ispc::vec4f*)clipPlaneData, numClipPlanes, _anaglyphEnabled,
                                 (ispc::vec3f&)_anaglyphIpdOffset);
}

AdvancedRenderer::AdvancedRenderer()
{
    ispcEquivalent = ::ispc::AdvancedRenderer_create(this);
}

OSP_REGISTER_RENDERER(AdvancedRenderer, advanced);
} // namespace ospray
} // namespace engine
} // namespace core