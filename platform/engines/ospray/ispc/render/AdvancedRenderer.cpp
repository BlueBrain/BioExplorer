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

#include <platform/core/common/geometry/Cone.h>
#include <platform/core/common/geometry/Cylinder.h>
#include <platform/core/common/geometry/SDFGeometry.h>
#include <platform/core/common/geometry/Sphere.h>

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/common/Model.h>
#include <ospray/SDK/geometry/Cylinders.h>
#include <ospray/SDK/geometry/Geometry.h>
#include <ospray/SDK/geometry/Spheres.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

// ispc exports
#include "AdvancedRenderer_ispc.h"

using namespace ospray;

extern "C"
{
    int AdvancedRenderer_getBytesPerPrimitive(const void* geometry)
    {
        const ospray::Geometry* base = static_cast<const ospray::Geometry*>(geometry);
        if (dynamic_cast<const ospray::Spheres*>(base))
            return sizeof(core::Sphere);
        else if (dynamic_cast<const ospray::Cylinders*>(base))
            return sizeof(core::Cylinder);
        else if (dynamic_cast<const ospray::Cones*>(base))
            return sizeof(core::Cone);
        else if (dynamic_cast<const ospray::SDFGeometries*>(base))
            return sizeof(core::SDFGeometry);
        return 0;
    }
}

namespace core
{
void AdvancedRenderer::commit()
{
    SimulationRenderer::commit();

    _alphaCorrection = getParam1f("alphaCorrection", 0.5f);
    _fogThickness = getParam1f("fogThickness", 1e6f);
    _fogStart = getParam1f("fogStart", 0.f);

    _exposure = getParam1f("mainExposure", 1.f);
    _epsilonFactor = getParam1f("epsilonFactor", 1.f);

    _maxBounces = getParam1i("maxBounces", 3);
    _randomNumber = rand() % 1000;

    _useHardwareRandomizer = getParam("useHardwareRandomizer", 0);
    _showBackground = getParam("showBackground", 0);

    _shadows = getParam1f("shadows", 0.f);
    _softShadows = getParam1f("softShadows", 0.f);
    _softShadowsSamples = getParam1i("softShadowsSamples", 1);

    _giStrength = getParam1f("giWeight", 0.f);
    _giDistance = getParam1f("giDistance", 1e20f);
    _giSamples = getParam1i("giSamples", 1);

    _matrixFilter = getParam("matrixFilter", 0);

    clipPlanes = getParamData("clipPlanes", nullptr);
    const auto clipPlaneData = clipPlanes ? clipPlanes->data : nullptr;
    const uint32 numClipPlanes = clipPlanes ? clipPlanes->numItems : 0;

    ispc::AdvancedRenderer_set(getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), _shadows, _softShadows,
                               _softShadowsSamples, _giStrength, _giDistance, _giSamples, _randomNumber, _timestamp,
                               spp, _lightPtr, _lightArray.size(), _exposure, _epsilonFactor, _fogThickness, _fogStart,
                               _useHardwareRandomizer, _maxBounces, _showBackground, _matrixFilter,
                               _simulationData ? (float*)_simulationData->data : nullptr, _simulationDataSize,
                               (const ispc::vec4f*)clipPlaneData, numClipPlanes);
}

AdvancedRenderer::AdvancedRenderer()
{
    ispcEquivalent = ispc::AdvancedRenderer_create(this);
}

OSP_REGISTER_RENDERER(AdvancedRenderer, advanced);
} // namespace core
