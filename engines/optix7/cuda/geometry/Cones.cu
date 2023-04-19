/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include <optix.h>

#include <sutil/vec_math.h>

#include "GeometryData.h"

namespace brayns
{
extern "C" __global__ void __intersection__cone()
{
    const GeometryData::Cone* cone =
        reinterpret_cast<GeometryData::Cone*>(optixGetSbtDataPointer());

    float3 v0 = cone->center;
    float3 v1 = cone->up;
    float radius0 = cone->centerRadius;
    float radius1 = cone->upRadius;

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    if (radius0 < radius1)
    {
        // swap radii and positions, so radius0 and v0 are always at the bottom
        const float tmpRadius = radius1;
        radius1 = radius0;
        radius0 = tmpRadius;

        const float3 tmpPos = v1;
        v1 = v0;
        v0 = tmpPos;
    }

    const float3 upVector = v1 - v0;
    const float upLength = length(upVector);

    // Compute the height of the full cone, in order to obtain its vertex
    const float deltaRadius = radius0 - radius1;
    const float tanA = deltaRadius / upLength;
    const float coneHeight = radius0 / tanA;
    const float squareTanA = tanA * tanA;
    const float div = sqrtf(1.f + squareTanA);
    if (div == 0.f)
        return;
    const float cosA = 1.f / div;

    const float3 V = v0 + normalize(upVector) * coneHeight;
    const float3 v = normalize(v0 - V);

    // Normal of the plane P determined by V and ray
    float3 n = normalize(cross(ray_dir, V - ray_orig));
    const float dotNV = dot(n, v);
    if (dotNV > 0.f)
        n = -n;

    const float squareCosTheta = 1.f - dotNV * dotNV;
    const float cosTheta = sqrtf(squareCosTheta);
    if (cosTheta < cosA)
        return; // no intersection

    if (squareCosTheta == 0.f)
        return;

    const float squareTanTheta = (1.f - squareCosTheta) / squareCosTheta;
    const float tanTheta = sqrtf(squareTanTheta);

    // Compute u-v-w coordinate system
    const float3 u = normalize(cross(v, n));
    const float3 w = normalize(cross(u, v));

    // Circle intersection of cone with plane P
    const float3 uComponent = sqrtf(squareTanA - squareTanTheta) * u;
    const float3 vwComponent = v + tanTheta * w;
    const float3 delta1 = vwComponent + uComponent;
    const float3 delta2 = vwComponent - uComponent;
    const float3 rayApex = V - ray_orig;

    const float3 normal1 = cross(ray_dir, delta1);
    const float length1 = length(normal1);

    if (length1 == 0.f)
        return;

    const float r1 = dot(cross(rayApex, delta1), normal1) / (length1 * length1);

    const float3 normal2 = cross(ray_dir, delta2);
    const float length2 = length(normal2);

    if (length2 == 0.f)
        return;

    const float r2 = dot(cross(rayApex, delta2), normal2) / (length2 * length2);

    float t_in = r1;
    float t_out = r2;
    if (r2 > 0.f)
    {
        if (r1 > 0.f)
        {
            if (r1 > r2)
            {
                t_in = r2;
                t_out = r1;
            }
        }
        else
            t_in = r2;
    }

    if (t_in > ray_tmin && t_in < ray_tmax)
    {
        const float3 p1 = ray_orig + t_in * ray_dir;
        // consider only the parts within the extents of the truncated cone
        if (dot(p1 - v1, v) > 0.f && dot(p1 - v0, v) < 0.f)
        {
            const float3 surfaceVec = normalize(p1 - V);
            const float3 normal = cross(cross(v, surfaceVec), surfaceVec);
            optixReportIntersection(t_in, 0, float3_as_uints(normal),
                                    __float_as_uint(r1));
            return;
        }
    }
    if (t_out > ray_tmin && t_out < ray_tmax)
    {
        const float3 p2 = ray_orig + t_out * ray_dir;
        // consider only the parts within the extents of the truncated cone
        if (dot(p2 - v1, v) > 0.f && dot(p2 - v0, v) < 0.f)
        {
            const float3 surfaceVec = normalize(p2 - V);
            const float3 normal = cross(cross(v, surfaceVec), surfaceVec);
            optixReportIntersection(t_out, 0, float3_as_uints(normal),
                                    __float_as_uint(r2));
            return;
        }
    }
}
} // namespace brayns
