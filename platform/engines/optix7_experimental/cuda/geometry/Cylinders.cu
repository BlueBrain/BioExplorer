/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <optix.h>

#include <sutil/vec_math.h>

#include "GeometryData.h"

namespace core
{
extern "C" __global__ void __intersection__cylinder()
{
    const GeometryData::Cylinder* cylinder =
        reinterpret_cast<GeometryData::Cylinder*>(optixGetSbtDataPointer());

    const float3 v0 = cylinder->center;
    const float3 v1 = cylinder->up;
    const float radius = cylinder->radius;

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    const float3 A = v0 - ray_orig;
    const float3 B = v1 - ray_orig;

    const float3 O = make_float3(0.f);
    const float3 V = ray_dir;

    const float3 AB = B - A;
    const float3 AO = O - A;

    const float3 AOxAB = cross(AO, AB);
    const float3 VxAB = cross(V, AB);
    const float ab2 = dot(AB, AB);
    const float a = dot(VxAB, VxAB);
    const float b = 2.f * dot(VxAB, AOxAB);
    const float c = dot(AOxAB, AOxAB) - (radius * radius * ab2);

    const float radical = b * b - 4.f * a * c;
    if (radical >= 0.f)
    {
        // clip to near and far cap of cylinder
        const float tA = dot(AB, A) / dot(V, AB);
        const float tB = dot(AB, B) / dot(V, AB);
        // const float tAB0 = max( 0.f, min( tA, tB ));
        // const float tAB1 = min( RT_DEFAULT_MAX, max( tA, tB ));
        const float tAB0 = min(tA, tB);
        const float tAB1 = max(tA, tB);

        const float srad = sqrt(radical);

        const float t = (-b - srad) / (2.f * a);

        bool check_second = true;
        float3 normal;
        if (t >= tAB0 && t <= tAB1 && t > ray_tmin && t < ray_tmax)
        {
            const float3 P = ray_orig + t * ray_dir - v0;
            const float3 V = cross(P, AB);
            normal = cross(AB, V);
            if (optixReportIntersection(t, 0, float3_as_uints(normal),
                                        __float_as_uint(radius)))
                check_second = false;
        }

        if (check_second)
        {
            const float t = (-b + srad) / (2.f * a);
            if (t >= tAB0 && t <= tAB1 && t > ray_tmin && t < ray_tmax)
            {
                const float3 P = t * ray_dir - A;
                const float3 V = cross(P, AB);
                normal = cross(AB, V);
                optixReportIntersection(t, 0, float3_as_uints(normal),
                                        __float_as_uint(radius));
            }
        }
    }
}
} // namespace core
