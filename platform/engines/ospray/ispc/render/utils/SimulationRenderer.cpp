/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

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

#include "SimulationRenderer.h"
#include "SimulationRenderer_ispc.h"

#include <platform/engines/ospray/ispc/geometry/Cones.h>
#include <platform/engines/ospray/ispc/geometry/SDFGeometries.h>

#include <platform/core/common/Properties.h>
#include <platform/core/common/geometry/Cone.h>
#include <platform/core/common/geometry/Cylinder.h>
#include <platform/core/common/geometry/SDFGeometry.h>
#include <platform/core/common/geometry/Sphere.h>

#include <ospray/SDK/geometry/Cylinders.h>
#include <ospray/SDK/geometry/Geometry.h>
#include <ospray/SDK/geometry/Spheres.h>
#include <ospray/SDK/transferFunction/TransferFunction.h>

extern "C"
{
    int SimulationRenderer_getBytesPerPrimitive(const void* geometry)
    {
        const ::ospray::Geometry* base = static_cast<const ::ospray::Geometry*>(geometry);
        if (dynamic_cast<const ::ospray::Spheres*>(base))
            return sizeof(core::Sphere);
        else if (dynamic_cast<const ::ospray::Cylinders*>(base))
            return sizeof(core::Cylinder);
        else if (dynamic_cast<const core::engine::ospray::Cones*>(base))
            return sizeof(core::Cone);
        else if (dynamic_cast<const core::engine::ospray::SDFGeometries*>(base))
            return sizeof(core::SDFGeometry);
        return 0;
    }
}

namespace core
{
namespace engine
{
namespace ospray
{
void SimulationRenderer::commit()
{
    AbstractRenderer::commit();

    _secondaryModel = (::ospray::Model*)getParamObject(RENDERER_PROPERTY_SECONDARY_MODEL, nullptr);
    _maxDistanceToSecondaryModel = getParam1f(RENDERER_PROPERTY_MAX_DISTANCE_TO_SECONDARY_MODEL.name.c_str(),
                                              DEFAULT_RENDERER_MAX_DISTANCE_TO_SECONDARY_MODEL);
    _userData = getParamData(RENDERER_PROPERTY_USER_DATA);
    _simulationDataSize = _userData ? _userData->size() : 0;
    _alphaCorrection = getParam1f(RENDERER_PROPERTY_ALPHA_CORRECTION.name.c_str(), DEFAULT_RENDERER_ALPHA_CORRECTION);
    _fogStart = getParam1f(RENDERER_PROPERTY_FOG_START.name.c_str(), DEFAULT_RENDERER_FOG_START);
    _fogThickness = getParam1f(RENDERER_PROPERTY_FOG_THICKNESS.name.c_str(), DEFAULT_RENDERER_FOG_THICKNESS);
    _exposure = getParam1f(COMMON_PROPERTY_EXPOSURE.name.c_str(), DEFAULT_COMMON_EXPOSURE);
    _timestamp = getParam1f(RENDERER_PROPERTY_TIMESTAMP.name.c_str(), DEFAULT_RENDERER_TIMESTAMP);
    _epsilonFactor = getParam1f(RENDERER_PROPERTY_EPSILON_MULTIPLIER.name.c_str(), DEFAULT_RENDERER_EPSILON_MULTIPLIER);
    _maxRayDepth = getParam1i(RENDERER_PROPERTY_MAX_RAY_DEPTH.name.c_str(), DEFAULT_RENDERER_MAX_RAY_DEPTH);
    _randomNumber = rand() % 1000;
    _useHardwareRandomizer = getParam(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER.name.c_str(),
                                      static_cast<int>(DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER));
    _showBackground = getParam(RENDERER_PROPERTY_SHOW_BACKGROUND.name.c_str(), DEFAULT_RENDERER_SHOW_BACKGROUND);

    // Transfer function
    ::ospray::TransferFunction* transferFunction =
        (::ospray::TransferFunction*)getParamObject(DEFAULT_COMMON_TRANSFER_FUNCTION, nullptr);
    if (transferFunction)
        ::ispc::SimulationRenderer_setTransferFunction(getIE(), transferFunction->getIE());
}
} // namespace ospray
} // namespace engine
} // namespace core