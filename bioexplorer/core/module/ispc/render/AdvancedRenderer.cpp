/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

// ospray
#include <ospray/SDK/common/Data.h>
#include <ospray/SDK/common/Model.h>
#include <ospray/SDK/lights/Light.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

#include <brayns/ispc/geometry/Cones.h>
#include <brayns/ispc/geometry/SDFGeometries.h>

#include <brayns/common/geometry/Cone.h>
#include <brayns/common/geometry/Cylinder.h>
#include <brayns/common/geometry/SDFGeometry.h>
#include <brayns/common/geometry/Sphere.h>

#include <ospray/SDK/geometry/Cylinders.h>
#include <ospray/SDK/geometry/Geometry.h>
#include <ospray/SDK/geometry/Spheres.h>

// ispc exports
#include "AdvancedRenderer_ispc.h"

using namespace ospray;

extern "C"
{
    int AdvancedRenderer_getBytesPerPrimitive(const void* geometry)
    {
        const ospray::Geometry* base =
            static_cast<const ospray::Geometry*>(geometry);
        if (dynamic_cast<const ospray::Spheres*>(base))
            return sizeof(brayns::Sphere);
        else if (dynamic_cast<const ospray::Cylinders*>(base))
            return sizeof(brayns::Cylinder);
        else if (dynamic_cast<const ospray::Cones*>(base))
            return sizeof(brayns::Cone);
        else if (dynamic_cast<const ospray::SDFGeometries*>(base))
            return sizeof(brayns::SDFGeometry);
        return 0;
    }
}

namespace bioexplorer
{
namespace rendering
{
void AdvancedRenderer::commit()
{
    SimulationRenderer::commit();

    _lightData = (ospray::Data*)getParamData("lights");
    _lightArray.clear();

    if (_lightData)
        for (size_t i = 0; i < _lightData->size(); ++i)
            _lightArray.push_back(
                ((ospray::Light**)_lightData->data)[i]->getIE());

    _lightPtr = _lightArray.empty() ? nullptr : &_lightArray[0];

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

    ispc::AdvancedRenderer_set(
        getIE(), (_bgMaterial ? _bgMaterial->getIE() : nullptr), _shadows,
        _softShadows, _softShadowsSamples, _giStrength, _giDistance, _giSamples,
        _randomNumber, _timestamp, spp, _lightPtr, _lightArray.size(),
        _exposure, _epsilonFactor, _fogThickness, _fogStart,
        _useHardwareRandomizer, _maxBounces, _showBackground, _matrixFilter,
        _simulationData ? (float*)_simulationData->data : nullptr,
        _simulationDataSize, (const ispc::vec4f*)clipPlaneData, numClipPlanes);
}

AdvancedRenderer::AdvancedRenderer()
{
    ispcEquivalent = ispc::AdvancedRenderer_create(this);
}

OSP_REGISTER_RENDERER(AdvancedRenderer, bio_explorer);
} // namespace rendering
} // namespace bioexplorer
