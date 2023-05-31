/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include <core/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

// ispc exports
#include "ProximityDetectionRenderer_ispc.h"

namespace sonataexplorer
{
void ProximityDetectionRenderer::commit()
{
    SimulationRenderer::commit();

    _nearColor = getParam3f("detectionNearColor", ospray::vec3f(0.f, 1.f, 0.f));
    _farColor = getParam3f("detectionFarColor", ospray::vec3f(1.f, 0.f, 0.f));
    _detectionDistance = getParam1f("detectionDistance", 1.f);
    _detectionOnDifferentMaterial = bool(getParam1i("detectionOnDifferentMaterial", 1));
    _surfaceShadingEnabled = bool(getParam1i("surfaceShadingEnabled", 1));
    _randomNumber = getParam1i("randomNumber", 0);
    _alphaCorrection = getParam1f("alphaCorrection", 0.5f);

    ispc::ProximityDetectionRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr),
                                         (ispc::vec3f&)_nearColor, (ispc::vec3f&)_farColor, _detectionDistance,
                                         _detectionOnDifferentMaterial, _randomNumber, _timestamp, spp,
                                         _surfaceShadingEnabled, _lightPtr, _lightArray.size(), _alphaCorrection,
                                         _maxBounces, _exposure, _useHardwareRandomizer);
}

ProximityDetectionRenderer::ProximityDetectionRenderer()
{
    ispcEquivalent = ispc::ProximityDetectionRenderer_create(this);
}

OSP_REGISTER_RENDERER(ProximityDetectionRenderer, proximity_detection);
OSP_REGISTER_MATERIAL(proximity_detection, AdvancedMaterial, default);
} // namespace sonataexplorer
