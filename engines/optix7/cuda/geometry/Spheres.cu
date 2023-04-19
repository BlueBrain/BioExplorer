/* Copyright (c) 2019, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
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

// #define TEST_AABB

namespace brayns
{
extern "C" __global__ void __intersection__sphere()
{
    // const int primID = optixGetPrimitiveIndex();
    const SphereHitGroupData* hit_group_data =
        reinterpret_cast<SphereHitGroupData*>(
            optixGetSbtDataPointer()); // + primID * sizeof(SphereHitGroupData);

#ifdef TEST_AABB
    optixReportIntersection(0, 0, float3_as_uints(make_float3(0, 0, -1)),
                            __float_as_uint(0.25f));
    return;
#endif

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float ray_tmin = optixGetRayTmin();
    const float ray_tmax = optixGetRayTmax();

    // const float3 center = make_float3(primID, 0.f, 0.f);
    const float3 center = hit_group_data->sphere.center;
    const float radius = hit_group_data->sphere.radius;

    const float3 O = ray_orig - center;
    const float l = 1.0f / length(ray_dir);
    const float3 D = ray_dir * l;

    float b = dot(O, D);
    float c = dot(O, O) - radius * radius;
    float disc = b * b - c;
    if (disc > 0.0f)
    {
        float sdisc = sqrtf(disc);
        float root1 = (-b - sdisc);
        float root11 = 0.0f;
        bool check_second = true;

        const bool do_refine = fabsf(root1) > (10.0f * radius);

        if (do_refine)
        {
            // refine root1
            float3 O1 = O + root1 * D;
            b = dot(O1, D);
            c = dot(O1, O1) - radius * radius;
            disc = b * b - c;

            if (disc > 0.0f)
            {
                sdisc = sqrtf(disc);
                root11 = (-b - sdisc);
            }
        }

        float t;
        float3 normal;
        t = (root1 + root11) * l;
        if (t > ray_tmin && t < ray_tmax)
        {
            normal = (O + (root1 + root11) * D) / radius;
            if (optixReportIntersection(t, 0, float3_as_uints(normal),
                                        __float_as_uint(radius)))
                check_second = false;
        }

        if (check_second)
        {
            float root2 = (-b + sdisc) + (do_refine ? root1 : 0);
            t = root2 * l;
            normal = (O + root2 * D) / radius;
            if (t > ray_tmin && t < ray_tmax)
                optixReportIntersection(t, 0, float3_as_uints(normal),
                                        __float_as_uint(radius));
        }
    }
}

} // namespace brayns
