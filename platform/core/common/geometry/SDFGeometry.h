/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
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

#pragma once

#include <platform/core/common/Types.h>

namespace core
{
enum class SDFType : uint8_t
{
    sdf_sphere = 0,
    sdf_pill = 1,
    sdf_cone_pill = 2,
    sdf_cone_pill_sigmoid = 3,
    sdf_cone = 4,
    sdf_torus = 5,
    sdf_cut_sphere = 6,
    sdf_vesica = 7
};

struct SDFGeometry
{
    uint64_t userData;
    Vector3f userParams; // Currently used exclusively for displacement
    Vector3f p0;
    Vector3f p1;
    float r0 = -1.f;
    float r1 = -1.f;
    uint64_t neighboursIndex = 0;
    uint8_t numNeighbours = 0;
    SDFType type;
};

inline SDFGeometry createSDFSphere(const Vector3f& center, const float radius, const uint64_t data = 0,
                                   const Vector3f& userParams = Vector3f(0.f))
{
    SDFGeometry geom;
    geom.userData = data;
    geom.userParams = userParams;
    geom.p0 = center;
    geom.r0 = radius;
    geom.type = SDFType::sdf_sphere;
    return geom;
}

inline SDFGeometry createSDFCutSphere(const Vector3f& center, const float radius, const float cutRadius,
                                      const uint64_t data = 0, const Vector3f& userParams = Vector3f(0.f))
{
    SDFGeometry geom;
    geom.userData = data;
    geom.userParams = userParams;
    geom.p0 = center;
    geom.r0 = radius;
    geom.r1 = cutRadius;
    geom.type = SDFType::sdf_cut_sphere;
    return geom;
}

inline SDFGeometry createSDFPill(const Vector3f& p0, const Vector3f& p1, const float radius, const uint64_t data = 0,
                                 const Vector3f& userParams = Vector3f(0.f))
{
    SDFGeometry geom;
    geom.userData = data;
    geom.userParams = userParams;
    geom.p0 = p0;
    geom.p1 = p1;
    geom.r0 = radius;
    geom.type = SDFType::sdf_pill;
    return geom;
}

inline SDFGeometry createSDFConePill(const Vector3f& p0, const Vector3f& p1, const float r0, const float r1,
                                     const uint64_t data = 0, const Vector3f& userParams = Vector3f(0.f))
{
    SDFGeometry geom;
    geom.userData = data;
    geom.userParams = userParams;
    geom.p0 = p0;
    geom.p1 = p1;
    geom.r0 = r0;
    geom.r1 = r1;

    if (r0 < r1)
    {
        std::swap(geom.p0, geom.p1);
        std::swap(geom.r0, geom.r1);
    }

    geom.type = SDFType::sdf_cone_pill;
    return geom;
}

inline SDFGeometry createSDFConePillSigmoid(const Vector3f& p0, const Vector3f& p1, const float r0, const float r1,
                                            const uint64_t data = 0, const Vector3f& userParams = Vector3f(0.f))
{
    SDFGeometry geom = createSDFConePill(p0, p1, r0, r1, data, userParams);
    geom.type = SDFType::sdf_cone_pill_sigmoid;
    return geom;
}

inline SDFGeometry createSDFTorus(const Vector3f& p0, const float r0, const float r1, const uint64_t data = 0,
                                  const Vector3f& userParams = Vector3f(0.f))
{
    SDFGeometry geom;
    geom.userData = data;
    geom.userParams = userParams;
    geom.p0 = p0;
    geom.r0 = r0;
    geom.r1 = r1;
    geom.type = SDFType::sdf_torus;
    return geom;
}

inline SDFGeometry createSDFVesica(const Vector3f& p0, const Vector3f& p1, const float r0, const uint64_t data = 0,
                                   const Vector3f& userParams = Vector3f(0.f))
{
    SDFGeometry geom;
    geom.userData = data;
    geom.userParams = userParams;
    geom.p0 = p0;
    geom.p1 = p1;
    geom.r0 = r0;
    geom.type = SDFType::sdf_vesica;
    return geom;
}

inline Boxd getSDFBoundingBox(const SDFGeometry& geom)
{
    Boxd bounds;
    switch (geom.type)
    {
    case SDFType::sdf_sphere:
    case SDFType::sdf_cut_sphere:
    {
        bounds.merge(geom.p0 - Vector3f(geom.r0));
        bounds.merge(geom.p0 + Vector3f(geom.r0));
        break;
    }
    case SDFType::sdf_pill:
    case SDFType::sdf_vesica:
    {
        bounds.merge(geom.p0 - Vector3f(geom.r0));
        bounds.merge(geom.p0 + Vector3f(geom.r0));
        bounds.merge(geom.p1 - Vector3f(geom.r0));
        bounds.merge(geom.p1 + Vector3f(geom.r0));
        break;
    }
    case SDFType::sdf_cone:
    case SDFType::sdf_cone_pill:
    case SDFType::sdf_cone_pill_sigmoid:
    {
        bounds.merge(geom.p0 - Vector3f(geom.r0));
        bounds.merge(geom.p0 + Vector3f(geom.r0));
        bounds.merge(geom.p1 - Vector3f(geom.r1));
        bounds.merge(geom.p1 + Vector3f(geom.r1));
        break;
    }
    case SDFType::sdf_torus:
    {
        bounds.merge(geom.p0 - Vector3f(geom.r0 + geom.r1));
        bounds.merge(geom.p0 + Vector3f(geom.r0 + geom.r1));
        break;
    }
    default:
        throw std::runtime_error("No bounds found for SDF type.");
    }
    return bounds;
}
} // namespace core