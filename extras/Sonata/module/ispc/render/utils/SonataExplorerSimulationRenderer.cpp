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

#include <common/Logs.h>

#include "SonataExplorerSimulationRenderer.h"
#include "SonataExplorerSimulationRenderer_ispc.h"

#include <brayns/ispc/geometry/Cones.h>
#include <brayns/ispc/geometry/SDFGeometries.h>

#include <brayns/common/geometry/Cone.h>
#include <brayns/common/geometry/Cylinder.h>
#include <brayns/common/geometry/SDFGeometry.h>
#include <brayns/common/geometry/Sphere.h>

#include <ospray/SDK/geometry/Cylinders.h>
#include <ospray/SDK/geometry/Geometry.h>
#include <ospray/SDK/geometry/Spheres.h>

#include <ospray/SDK/transferFunction/TransferFunction.h>

extern "C"
{
    int SonataExplorerSimulationRenderer_getBytesPerPrimitive(
        const void* geometry)
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

namespace sonataexplorer
{
void SonataExplorerSimulationRenderer::commit()
{
    SonataExplorerAbstractRenderer::commit();

    _secondaryModel = (ospray::Model*)getParamObject("secondaryModel", nullptr);
    _maxDistanceToSecondaryModel =
        getParam1f("maxDistanceToSecondaryModel", 30.f);

    _simulationData = getParamData("simulationData");
    _simulationDataSize = _simulationData ? _simulationData->size() : 0;

    _alphaCorrection = getParam1f("alphaCorrection", 0.5f);
    _fogThickness = getParam1f("fogThickness", 1e6f);
    _fogStart = getParam1f("fogStart", 0.f);

    // Transfer function
    ospray::TransferFunction* transferFunction =
        (ospray::TransferFunction*)getParamObject("transferFunction", nullptr);
    if (transferFunction)
        ispc::SonataExplorerSimulationRenderer_setTransferFunction(
            getIE(), transferFunction->getIE());
}

} // namespace sonataexplorer
