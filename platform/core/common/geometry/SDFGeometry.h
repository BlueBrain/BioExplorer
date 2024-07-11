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

#pragma once

#include "CommonDefines.h"

#include <platform/core/common/Types.h>

#include <Defines.h>

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
    sdf_vesica = 7,
    sdf_ellipsoid = 8
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
    __MEMORY_ALIGNMENT__
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

inline SDFGeometry createSDFEllipsoid(const Vector3f& p0, const Vector3f& r, const uint64_t data = 0,
                                      const Vector3f& userParams = Vector3f(0.f))
{
    SDFGeometry geom;
    geom.userData = data;
    geom.userParams = userParams;
    geom.p0 = p0;
    geom.p1 = r;
    geom.type = SDFType::sdf_ellipsoid;
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
    case SDFType::sdf_ellipsoid:
    {
        bounds.merge(geom.p0 - geom.p1);
        bounds.merge(geom.p0 + geom.p1);
        break;
    }
    default:
        throw std::runtime_error("No bounds found for SDF type.");
    }
    return bounds;
}
} // namespace core