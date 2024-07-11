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

#include "ProximityDetectionRenderer.h"

#include <common/Properties.h>

#include <platform/core/common/Properties.h>

#include <platform/core/common/Properties.h>
#include <platform/engines/ospray/OSPRayProperties.h>
#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

// ::ispc exports
#include "ProximityDetectionRenderer_ispc.h"

using namespace core;

namespace sonataexplorer
{
void ProximityDetectionRenderer::commit()
{
    SimulationRenderer::commit();

    _nearColor = getParam3f(SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_NEAR_COLOR.name.c_str(),
                            ::ospray::vec3f(SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_NEAR_COLOR[0],
                                            SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_NEAR_COLOR[1],
                                            SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_NEAR_COLOR[2]));
    _farColor = getParam3f(SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_FAR_COLOR.name.c_str(),
                           ::ospray::vec3f(SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_FAR_COLOR[0],
                                           SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_FAR_COLOR[1],
                                           SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_FAR_COLOR[2]));
    _detectionDistance = getParam1f(SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_DISTANCE.name.c_str(),
                                    SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_DISTANCE);
    _detectionOnDifferentMaterial =
        bool(getParam1i(SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_DIFFERENT_MATERIAL.name.c_str(),
                        SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_DIFFERENT_MATERIAL));
    _surfaceShadingEnabled =
        bool(getParam1i(SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_SURFACE_SHADING_ENABLED.name.c_str(),
                        SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_SURFACE_SHADING_ENABLED));
    _alphaCorrection = getParam1f(RENDERER_PROPERTY_ALPHA_CORRECTION.name.c_str(), DEFAULT_RENDERER_ALPHA_CORRECTION);
    ::ispc::ProximityDetectionRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr),
                                           (ispc::vec3f&)_nearColor, (ispc::vec3f&)_farColor, _detectionDistance,
                                           _detectionOnDifferentMaterial, _randomNumber, _timestamp, spp,
                                           _surfaceShadingEnabled, _lightPtr, _lightArray.size(), _alphaCorrection,
                                           _maxRayDepth, _exposure, _useHardwareRandomizer, _anaglyphEnabled,
                                           (ispc::vec3f&)_anaglyphIpdOffset);
}

ProximityDetectionRenderer::ProximityDetectionRenderer()
{
    ispcEquivalent = ::ispc::ProximityDetectionRenderer_create(this);
}

OSP_REGISTER_RENDERER(ProximityDetectionRenderer, proximity_detection);
OSP_REGISTER_MATERIAL(proximity_detection, core::engine::ospray::AdvancedMaterial, default);
} // namespace sonataexplorer
