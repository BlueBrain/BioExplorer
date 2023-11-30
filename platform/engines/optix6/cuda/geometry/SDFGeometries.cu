/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

// Ray-cone intersection: based on Ching-Kuang Shene (Graphics Gems 5, p. 227-230)

#include <platform/engines/optix6/cuda/Context.cuh>

// Global variables
rtDeclareVariable(uint, sdf_geometry_size, , );
rtBuffer<uint8_t> sdf_geometries_buffer;
rtBuffer<uint64_t> sdf_geometries_neighbours_buffer;

#define SDF_NO_INTERSECTION -1.f

enum SDFType : uint8_t
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

#define OFFSET_USER_DATA 0
#define OFFSET_USER_PARAMS (OFFSET_USER_DATA + sizeof(uint64_t))
#define OFFSET_P0 (OFFSET_USER_PARAMS + sizeof(float3))
#define OFFSET_P1 (OFFSET_P0 + sizeof(float3))
#define OFFSET_R0 (OFFSET_P1 + sizeof(float3))
#define OFFSET_R1 (OFFSET_R0 + sizeof(float))
#define OFFSET_NEIGHBOUR_INDEX (OFFSET_R1 + sizeof(float))
#define OFFSET_NB_NEIGHBOURS (OFFSET_NEIGHBOUR_INDEX + sizeof(uint64_t))
#define OFFSET_TYPE (OFFSET_NB_NEIGHBOURS + sizeof(uint8_t))

// NOTE: This layout must match exactly the 'SDFGeometry' struct in
// 'SDFGeometry.h'
struct SDFGeometry
{
    uint64_t userData;
    float3 userParams;
    float3 p0;
    float3 p1;
    float r0;
    float r1;
    uint64_t neighboursIndex;
    uint8_t numNeighbours;
    uint8_t type;
};

static __device__ inline float sign(const float x)
{
    return (x >= 0.f ? 1.f : -1.f);
}

static __device__ inline float lerp(const float factor, const float a, const float b)
{
    return (1.f - factor) * a + factor * b;
}

// polynomial smooth min (k = 0.1);
static __device__ inline float sminPoly(const float a, const float b, const float k)
{
    const float h = ::optix::clamp(0.5f + 0.5f * (b - a) / k, 0.f, 1.f);
    return lerp(h, b, a) - k * h * (1.f - h);
}

static __device__ inline float opDisplacement(const float3& p, const float a, const float b)
{
    return a * sin(b * p.x) * sin(b * p.y * 0.65f) * sin(b * p.z * 0.81f);
}

static __device__ inline float sdSphere(const float3& p, const float3& c, float r)
{
    return ::optix::length(p - c) - r;
}

static __device__ inline float sdCapsule(const float3& p, const float3& a, const float3& b, const float r)
{
    const float3 pa = p - a;
    const float3 ba = b - a;
    const float h = ::optix::clamp(::optix::dot(pa, ba) / ::optix::dot(ba, ba), 0.f, 1.f);
    return ::optix::length(pa - ba * h) - r;
}

static __device__ inline float sdConePill(const float3& p, const float3& a, const float3& b, const float r1,
                                          const float r2)
{
    // sampling independent computations (only depend on shape)
    const float3 ba = b - a;
    float l2 = ::optix::dot(ba, ba);
    float rr = r1 - r2;
    float a2 = l2 - rr * rr;
    float il2 = 1.0 / l2;

    // sampling dependant computations
    const float3 pa = p - a;
    const float y = ::optix::dot(pa, ba);
    const float z = y - l2;
    const float3 d = pa * l2 - ba * y;
    const float x2 = ::optix::dot(d, d);
    const float y2 = y * y * l2;
    const float z2 = z * z * l2;

    // single square root!
    const float k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k)
        return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k)
        return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

static __device__ inline float sdCone(const float3& p, const float3 a, const float3 b, float ra, float rb)
{
    float rba = rb - ra;
    float baba = ::optix::dot(b - a, b - a);
    float papa = ::optix::dot(p - a, p - a);
    float paba = ::optix::dot(p - a, b - a) / baba;

    float x = sqrt(papa - paba * paba * baba);

    float cax = max(0.0, x - ((paba < 0.5) ? ra : rb));
    float cay = abs(paba - 0.5) - 0.5;

    float k = rba * rba + baba;
    float f = ::optix::clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0);

    float cbx = x - ra - f * rba;
    float cby = paba - f;

    float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;

    return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}

static __device__ inline float sdTorus(const float3& p, const float3& c, float ra, float rb)
{
    float2 q = make_float2(::optix::length((make_float3(p.x, 0.f, p.z) - make_float3(c.x, 0.f, c.z))) - ra, p.y - c.y);
    return ::optix::length(q) - rb;
}

static __device__ inline float sdCutSphere(const float3& p, const float3 c, float r, float h)
{
    // sampling independent computations (only depend on shape)
    const float w = sqrt(r * r - h * h);

    // sampling dependant computations
    const float3 q =
        make_float3(::optix::length(make_float3(p.x, 0.f, p.z) - make_float3(c.x, 0.f, c.z)), p.y - c.y, 0.f);
    const float s = max((h - r) * q.x * q.x + w * w * (h + r - 2.f * q.y), h * q.x - w * q.y);
    return (s < 0.f) ? ::optix::length(q) - r : (q.x < w) ? h - q.y : ::optix::length(q - make_float3(w, h, 0.f));
}

static __device__ inline float sdVesica(const float3& p, const float3 a, const float3 b, float w)
{
    const float3 c = (a + b) * 0.5;
    float l = ::optix::length(b - a);
    const float3 v = (b - a) / l;
    float y = ::optix::dot(p - c, v);
    const float3 q = make_float3(::optix::length(p - c - y * v), abs(y), 0.f);

    const float r = 0.5f * l;
    const float d = 0.5f * (r * r - w * w) / w;
    const float3 h = (r * q.x < d * (q.y - r)) ? make_float3(0.f, r, 0.f) : make_float3(-d, 0.f, d + w);

    return ::optix::length(q - make_float3(h.x, h.y, 0.f)) - h.z;
}

static __device__ inline bool intersectBox(const ::optix::Aabb& box, float& t0, float& t1)
{
    const float3 a = (box.m_min - ray.origin) / ray.direction;
    const float3 b = (box.m_max - ray.origin) / ray.direction;
    const float3 near = fminf(a, b);
    const float3 far = fmaxf(a, b);
    t0 = fmaxf(near);
    t1 = fminf(far);

    return (t0 <= t1);
}

static __device__ inline SDFGeometry* getPrimitive(const int primIdx)
{
    const uint64_t bufferIndex = primIdx * sdf_geometry_size;
    return (SDFGeometry*)&sdf_geometries_buffer[bufferIndex];
}

static __device__ inline uint64_t getNeighbourIndex(const uint8_t index)
{
    return sdf_geometries_neighbours_buffer[index];
}

static __device__ inline ::optix::Aabb getBounds(const SDFGeometry* primitive)
{
    const float radius = max(primitive->r0, primitive->r1) + primitive->userParams.x;
    ::optix::Aabb aabb;
    switch (primitive->type)
    {
    case SDFType::sdf_sphere:
    case SDFType::sdf_cut_sphere:
    {
        aabb.m_min = primitive->p0 - radius;
        aabb.m_max = primitive->p0 + radius;
        return aabb;
    }
    case SDFType::sdf_torus:
    {
        const float r = primitive->r0 + primitive->r1 + primitive->userParams.x;
        aabb.m_min = primitive->p0 - r;
        aabb.m_max = primitive->p0 + r;
        return aabb;
    }
    default:
        aabb.m_min = make_float3(min(primitive->p0.x, primitive->p1.x), min(primitive->p0.y, primitive->p1.y),
                                 min(primitive->p0.z, primitive->p1.z)) -
                     radius;
        aabb.m_max = make_float3(max(primitive->p0.x, primitive->p1.x), max(primitive->p0.y, primitive->p1.y),
                                 max(primitive->p0.z, primitive->p1.z)) +
                     radius;
        return aabb;
    }
}

static __device__ inline float calcDistance(const SDFGeometry* primitive, const float3& position,
                                            const bool processDisplacement)
{
    const float displacement = (processDisplacement && primitive->userParams.x > 0.f)
                                   ? opDisplacement(position, primitive->userParams.x, primitive->userParams.y)
                                   : 0.f;
    switch (primitive->type)
    {
    case SDFType::sdf_sphere:
        return displacement + sdSphere(position, primitive->p0, primitive->r0);
    case SDFType::sdf_pill:
        return displacement + sdCapsule(position, primitive->p0, primitive->p1, primitive->r0);
    case SDFType::sdf_cone_pill:
    case SDFType::sdf_cone_pill_sigmoid:
        return displacement + sdConePill(position, primitive->p0, primitive->p1, primitive->r0, primitive->r1);
    case SDFType::sdf_cone:
        return displacement + sdCone(position, primitive->p0, primitive->p1, primitive->r0, primitive->r1);
    case SDFType::sdf_torus:
        return displacement + sdTorus(position, primitive->p0, primitive->r0, primitive->r1);
    case SDFType::sdf_cut_sphere:
        return displacement + sdCutSphere(position, primitive->p0, primitive->r0, primitive->r1);
    case SDFType::sdf_vesica:
        return displacement + sdVesica(position, primitive->p0, primitive->p1, primitive->r0);
    }
    return SDF_NO_INTERSECTION; // TODO: Weird return value...
}

static __device__ inline float sdfDistance(const float3& position, const SDFGeometry* primitive,
                                           const bool processDisplacement)
{
    float distance = calcDistance(primitive, position, processDisplacement);
    if (processDisplacement && primitive->numNeighbours > 0)
    {
        const float r0 = max(primitive->r0, primitive->r1);
        for (uint8_t i = 0; i < primitive->numNeighbours; ++i)
        {
            const uint64_t neighbourIndex = getNeighbourIndex(i);
            const SDFGeometry* neighbourGeometry = getPrimitive(neighbourIndex);
            const float neighbourDistance = calcDistance(neighbourGeometry, position, processDisplacement);
            if (neighbourDistance < 0.f)
                continue;
            const float r1 = max(neighbourGeometry->r0, neighbourGeometry->r1);
            const float blendFactor = lerp(geometrySdfBlendLerpFactor, min(r0, r1), max(r0, r1));
            distance = sminPoly(neighbourDistance, distance, blendFactor * geometrySdfBlendFactor);
        }
    }
    return distance;
}

static __device__ inline float3 computeNormal(const float3& position, const SDFGeometry* primitive,
                                              const bool processDisplacement)
{
    // tetrahedron technique (4 evaluations)
    const float t = 0.1f;
    const float3 k0 = make_float3(t, -t, -t);
    const float3 k1 = make_float3(-t, -t, t);
    const float3 k2 = make_float3(-t, t, -t);
    const float3 k3 = make_float3(t, t, t);
    return ::optix::normalize(k0 * sdfDistance(position + geometrySdfEpsilon * k0, primitive, processDisplacement) +
                              k1 * sdfDistance(position + geometrySdfEpsilon * k1, primitive, processDisplacement) +
                              k2 * sdfDistance(position + geometrySdfEpsilon * k2, primitive, processDisplacement) +
                              k3 * sdfDistance(position + geometrySdfEpsilon * k3, primitive, processDisplacement));
}

static __device__ inline float rayMarching(const SDFGeometry* primitive, bool& processDisplacement)
{
    const ::optix::Aabb box = getBounds(primitive);

    float t0, t1;
    if (!intersectBox(box, t0, t1))
        return SDF_NO_INTERSECTION;

    // TODO compute pixel radius
    const float pixel_radius = geometrySdfEpsilon;

    float omega = geometrySdfOmega;
    float t = t0;
    float candidateError = 1e6f;
    float tCandidate = t0;
    float previousRadius = 0.f;
    float stepLength = 0.f;
    uint64_t stepCount = 0;
    const bool forceHit = true;

    // check if we start inside or outside of the shape
    const float sdfSign = (sdfDistance(ray.origin, primitive, true) < 0.f ? -1 : 1);

    for (uint64_t i = 0; i < geometrySdfNbMarchIterations; i++)
    {
        const float3 p = ray.origin + ray.direction * t;
        processDisplacement = (/*ray.flags == RAY_FLAG_PRIMARY && */ ::optix::length(p - eye) < geometrySdfDistance);

        float signed_radius = sdfSign * sdfDistance(p, primitive, processDisplacement);
        float radius = abs(signed_radius);
        bool sorFail = (omega > 1.f && (radius + previousRadius) < stepLength);

        if (sorFail)
        {
            stepLength -= omega * stepLength;
            omega = 1.f;
        }
        else
            stepLength = signed_radius * omega;

        previousRadius = radius;
        float error = radius / t;
        if (!sorFail && error < candidateError)
        {
            tCandidate = t;
            candidateError = error;
        }

        if (!sorFail && (error < pixel_radius || t > t1))
            break;

        t += stepLength;
        ++stepCount;
    }

    if (t > t1 || (candidateError > pixel_radius && !forceHit))
        return SDF_NO_INTERSECTION;

    return tCandidate;
}

template <bool use_robust_method>
static __device__ void intersect_sdf_geometry(int primIdx)
{
    const SDFGeometry* primitive = getPrimitive(primIdx);
    bool processDisplacement = true;
    const float t_in = rayMarching(primitive, processDisplacement);

    if (t_in > 0.f)
    {
        if (rtPotentialIntersection(t_in))
            if (t_in > ray.tmin && t_in < ray.tmax)
            {
                const float3 position = ray.origin + t_in * ray.direction;
                shading_normal = geometric_normal = computeNormal(position, primitive, processDisplacement);
                userDataIndex = primitive->userData;
                texcoord = make_float2(0.f);
                texcoord3d = make_float3(0.f);
                rtReportIntersection(0);
            }
    }
}

RT_PROGRAM void intersect(int primIdx)
{
    intersect_sdf_geometry<false>(primIdx);
}

RT_PROGRAM void robust_intersect(int primIdx)
{
    intersect_sdf_geometry<true>(primIdx);
}

RT_PROGRAM void bounds(int primIdx, float result[6])
{
    const SDFGeometry* primitive = getPrimitive(primIdx);
    const ::optix::Aabb bounds = getBounds(primitive);
    memcpy(&result[0], &bounds[0], sizeof(optix::Aabb));
}
