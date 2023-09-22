/*
 * Copyright (c) 2018, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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
void SimulationRenderer::commit()
{
    AbstractRenderer::commit();

    _secondaryModel = (ospray::Model*)getParamObject(RENDERER_PROPERTY_SECONDARY_MODEL, nullptr);
    _maxDistanceToSecondaryModel = getParam1f(RENDERER_PROPERTY_MAX_DISTANCE_TO_SECONDARY_MODEL.name.c_str(),
                                              DEFAULT_RENDERER_MAX_DISTANCE_TO_SECONDARY_MODEL);
    _userData = getParamData(RENDERER_PROPERTY_USER_DATA);
    _simulationDataSize = _userData ? _userData->size() : 0;
    _alphaCorrection = getParam1f(RENDERER_PROPERTY_ALPHA_CORRECTION.name.c_str(), DEFAULT_RENDERER_ALPHA_CORRECTION);
    _fogStart = getParam1f(RENDERER_PROPERTY_FOG_START.name.c_str(), DEFAULT_RENDERER_FOG_START);
    _fogThickness = getParam1f(RENDERER_PROPERTY_FOG_THICKNESS.name.c_str(), DEFAULT_RENDERER_FOG_THICKNESS);
    _exposure = getParam1f(COMMON_PROPERTY_EXPOSURE.name.c_str(), DEFAULT_COMMON_EXPOSURE);
    _timestamp = getParam1f(RENDERER_PROPERTY_TIMESTAMP, DEFAULT_RENDERER_TIMESTAMP);
    _epsilonFactor = getParam1f(RENDERER_PROPERTY_EPSILON_MULTIPLIER.name.c_str(), DEFAULT_RENDERER_EPSILON_MULTIPLIER);
    _maxBounces = getParam1i(RENDERER_PROPERTY_MAX_RAY_DEPTH.name.c_str(), DEFAULT_RENDERER_MAX_RAY_DEPTH);
    _randomNumber = rand() % 1000;
    _useHardwareRandomizer = getParam(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER.name.c_str(),
                                      static_cast<int>(DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER));
    _showBackground = getParam(RENDERER_PROPERTY_SHOW_BACKGROUND.name.c_str(), DEFAULT_RENDERER_SHOW_BACKGROUND);

    // Transfer function
    ospray::TransferFunction* transferFunction =
        (ospray::TransferFunction*)getParamObject(RENDERER_PROPERTY_TRANSFER_FUNCTION, nullptr);
    if (transferFunction)
        ispc::SimulationRenderer_setTransferFunction(getIE(), transferFunction->getIE());
}
} // namespace core
