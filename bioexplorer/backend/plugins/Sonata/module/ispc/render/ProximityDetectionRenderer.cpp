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
    _randomNumber = getParam1i(core::RENDERER_PROPERTY_RANDOM_NUMBER, 0);
    _alphaCorrection = getParam1f(RENDERER_PROPERTY_ALPHA_CORRECTION.name.c_str(), DEFAULT_RENDERER_ALPHA_CORRECTION);

    ::ispc::ProximityDetectionRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr),
                                           (ispc::vec3f&)_nearColor, (ispc::vec3f&)_farColor, _detectionDistance,
                                           _detectionOnDifferentMaterial, _randomNumber, _timestamp, spp,
                                           _surfaceShadingEnabled, _lightPtr, _lightArray.size(), _alphaCorrection,
                                           _maxBounces, _exposure, _useHardwareRandomizer, _anaglyphEnabled,
                                           (ispc::vec3f&)_anaglyphIpdOffset);
}

ProximityDetectionRenderer::ProximityDetectionRenderer()
{
    ispcEquivalent = ::ispc::ProximityDetectionRenderer_create(this);
}

OSP_REGISTER_RENDERER(ProximityDetectionRenderer, proximity_detection);
OSP_REGISTER_MATERIAL(proximity_detection, core::engine::ospray::AdvancedMaterial, default);
} // namespace sonataexplorer
