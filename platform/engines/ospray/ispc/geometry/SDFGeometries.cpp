/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include <platform/core/common/geometry/SDFGeometry.h>
#include <platform/engines/ospray/OSPRayProperties.h>

#include "SDFGeometries.h"
#include "ospray/SDK/common/Data.h"
#include "ospray/SDK/common/Model.h"

#include "SDFGeometries_ispc.h"

#include <climits>
#include <cstddef>

namespace core
{
namespace engine
{
namespace ospray
{
SDFGeometries::SDFGeometries()
{
    this->ispcEquivalent = ::ispc::SDFGeometries_create(this);
}

void SDFGeometries::finalize(::ospray::Model* model)
{
    data = getParamData(OSPRAY_GEOMETRY_PROPERTY_SDF, nullptr);
    neighbours = getParamData(OSPRAY_GEOMETRY_PROPERTY_SDF_NEIGHBOURS, nullptr);
    geometries = getParamData(OSPRAY_GEOMETRY_PROPERTY_SDF_GEOMETRIES, nullptr);
    epsilon = getParam1f(OSPRAY_GEOMETRY_PROPERTY_SDF_EPSILON, DEFAULT_GEOMETRY_SDF_EPSILON);
    nbMarchIterations =
        getParam1i(OSPRAY_GEOMETRY_PROPERTY_SDF_NB_MARCH_ITERATIONS, DEFAULT_GEOMETRY_SDF_NB_MARCH_ITERATIONS);
    blendFactor = getParam1f(OSPRAY_GEOMETRY_PROPERTY_SDF_BLEND_FACTOR, DEFAULT_GEOMETRY_SDF_BLEND_FACTOR);
    blendLerpFactor =
        getParam1f(OSPRAY_GEOMETRY_PROPERTY_SDF_BLEND_LERP_FACTOR, DEFAULT_GEOMETRY_SDF_BLEND_LERP_FACTOR);
    omega = getParam1f(OSPRAY_GEOMETRY_PROPERTY_SDF_OMEGA, DEFAULT_GEOMETRY_SDF_OMEGA);
    distance = getParam1f(OSPRAY_GEOMETRY_PROPERTY_SDF_DISTANCE, DEFAULT_GEOMETRY_SDF_DISTANCE);

    if (data.ptr == nullptr)
        throw std::runtime_error(
            "#ospray:geometry/SDFGeometries: "
            "no 'sdfgeometries' data specified");

    const size_t numSDFGeometries = data->numItems;
    const size_t numNeighbours = neighbours->numItems;

    bounds = ::ospray::empty;
    const auto geoms = static_cast<core::SDFGeometry*>(geometries->data);
    for (size_t i = 0; i < numSDFGeometries; i++)
    {
        const auto bd = getSDFBoundingBox(geoms[i]);
        const auto& bMind = bd.getMin();
        const auto& bMaxd = bd.getMax();
        const auto bMinf = ::ospray::vec3f(bMind[0], bMind[1], bMind[2]);
        const auto bMaxf = ::ospray::vec3f(bMaxd[0], bMaxd[1], bMaxd[2]);

        bounds.extend(bMinf);
        bounds.extend(bMaxf);
    }

    ::ispc::SDFGeometriesGeometry_set(getIE(), model->getIE(), data->data, numSDFGeometries, neighbours->data,
                                      numNeighbours, geometries->data, epsilon, nbMarchIterations, blendFactor,
                                      blendLerpFactor, omega, distance);
}

OSP_REGISTER_GEOMETRY(SDFGeometries, sdfgeometries);
} // namespace ospray
} // namespace engine
} // namespace core