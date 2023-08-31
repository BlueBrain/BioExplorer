/*
 * Copyright (c) 2019, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
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

#include <platform/engines/optix6/cuda/Context.cuh>

using namespace optix;

// Global variables
rtBuffer<float4> vertices_buffer;
rtBuffer<int> indices_buffer;

static __device__ void intersect_sphere(const float3& center, float& t_in, float& t_out, const float r)
{
    const float rd2 = 1.0f / dot(ray.direction, ray.direction);
    const float3 CO = center - ray.origin;
    const float projCO = dot(CO, ray.direction) * rd2;
    const float3 perp = CO - projCO * ray.direction;
    const float l2 = dot(perp, perp);
    const float r2 = r * r;
    if (l2 > r2)
        return;

    const float td = sqrt((r2 - l2) * rd2);
    const float sph_t_in = projCO - td;
    const float sph_t_out = projCO + td;

    t_in = min(t_in, sph_t_in);
    t_out = max(t_out, sph_t_out);
}

static __device__ void intersect_cylinder(const float3& v0, const float3& v1, float& t_in, float& t_out, const float r)
{
    const float3 d = ray.direction;
    const float3 s = v1 - v0;
    const float3 sxd = cross(s, d);
    const float a = dot(sxd, sxd);
    if (a == 0.f)
        return;

    const float3 f = v0 - ray.origin;
    const float3 sxf = cross(s, f);
    const float ra = 1.f / a;
    const float ts = dot(sxd, sxf) * ra;
    const float3 fp = f - ts * d;

    const float s2 = dot(s, s);
    const float3 perp = cross(s, fp);
    const float c = r * r * s2 - dot(perp, perp);
    if (c < 0.f)
        return;

    float td = sqrt(c * ra);
    const float tin = ts - td;
    const float tout = ts + td;

    const float rsd = 1.f / dot(s, d);
    const float tA = dot(s, f) * rsd;
    const float tB = tA + s2 * rsd;
    const float cyl_t_in = max(min(tA, tB), tin);
    const float cyl_t_out = min(max(tA, tB), tout);

    if (cyl_t_in < cyl_t_out)
    {
        t_in = cyl_t_in;
        t_out = cyl_t_out;
    }
}

template <bool use_robust_method>
static __device__ void intersect_streamlines(int primIdx)
{
    const int idx = indices_buffer[primIdx];
    const float radius = vertices_buffer[idx].w;
    const float3 A = make_float3(vertices_buffer[idx]);
    const float3 B = make_float3(vertices_buffer[idx + 1]);

    float t_in = INFINITY;
    float t_out = -INFINITY;
    intersect_cylinder(A, B, t_in, t_out, radius);
    intersect_sphere(A, t_in, t_out, radius);
    intersect_sphere(B, t_in, t_out, radius);

    bool hit = false;
    float t = t_hit;
    if (t_in < t_out)
    {
        if (t_in > ray.tmin && t_in < t_hit)
        {
            t = t_in;
            hit = true;
        }
        else if (t_out > ray.tmin && t_out < t_hit)
        {
            t = t_out;
            hit = true;
        }
    }

    if (hit)
    {
        if (rtPotentialIntersection(t))
        {
            const float3 p = ray.origin + t * ray.direction;
            float s = dot(p - A, B - A) * (1.f / dot(B - A, B - A));
            s = min(max(s, 0.f), 1.f);
            const float3 PonAxis = A + s * (B - A);
            simulation_idx = 0;
            texcoord = make_float2(float(primIdx % MAX_TEXTURE_SIZE) / float(MAX_TEXTURE_SIZE),
                                   float(uint(primIdx / MAX_TEXTURE_SIZE)) /
                                       (float(vertices_buffer.size()) / float(MAX_TEXTURE_SIZE)));
            texcoord3d = make_float3(0.f);
            geometric_normal = shading_normal = normalize(p - PonAxis);
            rtReportIntersection(0);
        }
    }
}

RT_PROGRAM void intersect(int primIdx)
{
    intersect_streamlines<false>(primIdx);
}

RT_PROGRAM void robust_intersect(int primIdx)
{
    intersect_streamlines<true>(primIdx);
}

static __device__ float3 min3(const float3& a, const float3& b)
{
    return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

static __device__ float3 max3(const float3& a, const float3& b)
{
    return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

RT_PROGRAM void bounds(int primIdx, float result[6])
{
    const int idx = indices_buffer[primIdx];
    const float radius = max(vertices_buffer[idx].w, vertices_buffer[idx + 1].w);
    const float3 A = make_float3(vertices_buffer[idx]);
    const float3 B = make_float3(vertices_buffer[idx + 1]);
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = min3(A, B) - radius;
    aabb->m_max = max3(A, B) + radius;
}
