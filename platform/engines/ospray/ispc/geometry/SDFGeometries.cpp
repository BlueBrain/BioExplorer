/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Jonas Karlsson <jonas.karlsson@epfl.ch>
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