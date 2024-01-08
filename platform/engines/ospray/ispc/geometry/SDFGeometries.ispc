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

// ospray
#include "ospray/SDK/common/Model.ih"
#include "ospray/SDK/common/Ray.ih"
#include "ospray/SDK/geometry/Geometry.ih"
#include "ospray/SDK/math/box.ih"
#include "ospray/SDK/math/vec.ih"
#include "ospray/version.h"

// embree
#include "embree3/rtcore.isph"
#include "embree3/rtcore_geometry.isph"
#include "embree3/rtcore_scene.isph"
#include "utils/SafeIncrement.ih"

#include "RayMarching.isph"

#define SDF_NO_INTERSECTION -1.f

enum SDFType
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

inline vec3f mod(const vec3f& v, const int m)
{
    return make_vec3f(v.x - m * floor(v.x / m), v.y - m * floor(v.y / m), v.z - m * floor(v.z / m));
}

inline vec3f mod(const vec3f& v, const vec3f& u)
{
    return v - u * floor(v / u);
}

// https://en.wikipedia.org/wiki/Smoothstep
inline float smoothstep(const float x)
{
    return x * x * (3 - 2 * x);
}

// https://en.wikipedia.org/wiki/Smoothstep
inline float smootherStep(const float x)
{
    return x * x * x * (x * (x * 6 - 15) + 10);
}

inline float opDisplacement(const vec3f& p, const float amplitude, const float frequency)
{
    return amplitude *
           (0.7f * sin(frequency * p.x * 0.72f) * sin(frequency * p.y * 0.65f) * sin(frequency * p.z * 0.81f) +
            0.3f * cos(p.x * 2.12f) * cos(p.y * 2.23f) * cos(p.z * 2.41f));
}

inline float sdSphere(const vec3f& p, const vec3f c, float r)
{
    return length(p - c) - r;
}

inline float sdCapsule(const vec3f& p, const vec3f a, const vec3f b, float r)
{
    const vec3f pa = p - a, ba = b - a;
    const float h = clamp(dot(pa, ba) / dot(ba, ba), 0.f, 1.f);
    return length(pa - ba * h) - r;
}

// polynomial smooth min (k = 0.1);
inline float sminPoly(const float a, const float b, const float k)
{
    const float h = clamp(0.5f + 0.5f * (b - a) / k, 0.f, 1.f);
    return lerp(h, b, a) - k * h * (1.f - h);
}

inline float smin(const float a, const float b, const float k)
{
    float res = exp(-k * a) + exp(-k * b);
    return -log(res) / k;
}

inline float opBlend(const float a, const float b)
{
    return smin(a, b, 5);
}

inline float opS(const float d1, const float d2)
{
    return max(-d2, d1);
}

inline float opU(const float a, const float b)
{
    return (a < b) ? a : b;
}

inline vec3f opRep(const vec3f& p, const vec3f& c)
{
    return mod(p, c) - 0.5f * c;
}

inline float mix(const float x, const float y, const float a)
{
    return x * (1.f - a) + y * a;
}

inline float sign(const float x)
{
    return (x >= 0.f ? 1.f : -1.f);
}

inline float sdConePill(const vec3f& p, const vec3f a, const vec3f b, const float r1, const float r2,
                        const bool /*useSigmoid*/)
{
    // sampling independent computations (only depend on shape)
    const vec3f ba = b - a;
    float l2 = dot(ba, ba);
    float rr = r1 - r2;
    float a2 = l2 - rr * rr;
    float il2 = 1.0 / l2;

    // sampling dependant computations
    const vec3f pa = p - a;
    const float y = dot(pa, ba);
    const float z = y - l2;
    const vec3f d = pa * l2 - ba * y;
    const float x2 = dot(d, d);
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

inline float sdCone(const vec3f& p, const vec3f a, const vec3f b, float ra, float rb)
{
    float rba = rb - ra;
    float baba = dot(b - a, b - a);
    float papa = dot(p - a, p - a);
    float paba = dot(p - a, b - a) / baba;

    float x = sqrt(papa - paba * paba * baba);

    float cax = max(0.0, x - ((paba < 0.5) ? ra : rb));
    float cay = abs(paba - 0.5) - 0.5;

    float k = rba * rba + baba;
    float f = clamp((rba * (x - ra) + paba * baba) / k, 0.0, 1.0);

    float cbx = x - ra - f * rba;
    float cby = paba - f;

    float s = (cbx < 0.f && cay < 0.f) ? -1.f : 1.f;

    return s * sqrt(min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));
}

inline float sdTorus(const vec3f& p, const vec3f c, float ra, float rb)
{
    const vec3f q = make_vec3f(length((make_vec3f(p.x, 0.f, p.z) - c)) - ra, p.y - c.y, 0.f);
    return length(q) - rb;
}

inline float sdCutSphere(const vec3f& p, const vec3f c, float r, float h)
{
    // sampling independent computations (only depend on shape)
    const float w = sqrt(r * r - h * h);

    // sampling dependant computations
    const vec3f q = make_vec3f(length(make_vec3f(p.x, 0.f, p.z) - make_vec3f(c.x, 0.f, c.z)), p.y - c.y, 0.f);
    const float s = max((h - r) * q.x * q.x + w * w * (h + r - 2.f * q.y), h * q.x - w * q.y);
    return (s < 0.f) ? length(q) - r : (q.x < w) ? h - q.y : length(q - make_vec3f(w, h, 0.));
}

inline float sdVesica(const vec3f& p, const vec3f a, const vec3f b, float w)
{
    const vec3f c = (a + b) * 0.5;
    float l = length(b - a);
    const vec3f v = (b - a) / l;
    float y = dot(p - c, v);
    const vec3f q = make_vec3f(length(p - c - y * v), abs(y), 0.f);

    const float r = 0.5f * l;
    const float d = 0.5f * (r * r - w * w) / w;
    const vec3f h = (r * q.x < d * (q.y - r)) ? make_vec3f(0.f, r, 0.f) : make_vec3f(-d, 0.f, d + w);

    return length(q - make_vec3f(h.x, h.y, 0.f)) - h.z;
}

// NOTE: This layout must match exactly the 'SDFGeometry' struct in
// 'SDFGeometry.h'
struct SDFGeometry
{
    uint64 userData;
    vec3f userParams;
    vec3f p0;
    vec3f p1;
    float r0;
    float r1;
    uint64 neighboursIndex;
    uint8 numNeighbours;
    uint8 type;
};

struct SDFGeometries
{
    uniform Geometry super;

    uniform SDFGeometry* uniform geometries;
    uniform uint64* uniform geometryRefs;
    uniform uint64* uniform neighbours;

    uniform bool useSafeIncrement;
    uniform float epsilon;
    uniform uint64 nbMarchIterations;
    uniform float blendFactor;
    uniform float blendLerpFactor;
    uniform float omega;
    uniform float distance;
};

typedef uniform SDFGeometries* uniform Geo_ptr;
typedef uniform SDFGeometry* uniform Prim_ptr;

DEFINE_SAFE_INCREMENT(SDFGeometry);
DEFINE_SAFE_INCREMENT(uint64);

uniform uint64 primToIdx(const uniform SDFGeometries& geometry, const uniform uint64 primID)
{
    return *safeIncrement(geometry.useSafeIncrement, geometry.geometryRefs, primID);
}

varying uint64 primToIdxVarying(const uniform SDFGeometries& geometry, const varying uint64 primID)
{
    return *safeIncrement(geometry.useSafeIncrement, geometry.geometryRefs, primID);
}

uniform uint64 getNeighbourIdx(const uniform SDFGeometries& geometry, const uniform uint64 startIdx,
                               const uniform uint64 neighIdx)
{
    return *safeIncrement(geometry.useSafeIncrement, geometry.neighbours, startIdx + neighIdx);
}

const Prim_ptr getPrimitive(const uniform SDFGeometries& geometry, const uniform uint64 idx)
{
    return safeIncrement(geometry.useSafeIncrement, geometry.geometries, idx);
}

const uniform SDFGeometry* varying getPrimitiveVarying(const uniform SDFGeometries& geometry, const varying uint64 idx)
{
    return safeIncrement(geometry.useSafeIncrement, geometry.geometries, idx);
}

inline float calcDistance(const uniform SDFGeometry& primitive, const vec3f& p, const bool processDisplacement)
{
    const float displacement = (processDisplacement && primitive.userParams.x > 0.f)
                                   ? opDisplacement(p, primitive.userParams.x, primitive.userParams.y)
                                   : 0.f;
    switch (primitive.type)
    {
    case sdf_sphere:
        return displacement + sdSphere(p, primitive.p0, primitive.r0);
    case sdf_pill:
        return displacement + sdCapsule(p, primitive.p0, primitive.p1, primitive.r0);
    case sdf_cone_pill:
        return displacement + sdConePill(p, primitive.p0, primitive.p1, primitive.r0, primitive.r1, false);
    case sdf_cone_pill_sigmoid:
        return displacement + sdConePill(p, primitive.p0, primitive.p1, primitive.r0, primitive.r1, true);
    case sdf_cone:
        return displacement + sdCone(p, primitive.p0, primitive.p1, primitive.r0, primitive.r1);
    case sdf_torus:
        return displacement + sdTorus(p, primitive.p0, primitive.r0, primitive.r1);
    case sdf_cut_sphere:
        return displacement + sdCutSphere(p, primitive.p0, primitive.r0, primitive.r1);
    case sdf_vesica:
        return displacement + sdVesica(p, primitive.p0, primitive.p1, primitive.r0);
    }
    return SDF_NO_INTERSECTION;
}

uniform box3fa sdfBounds(uDataPtr_t geo_, uDataPtr_t prim_)
{
    const Prim_ptr prim = (const Prim_ptr)prim_;

    const uniform float radius = max(prim->r0, prim->r1) + prim->userParams.x;
    switch (prim->type)
    {
    case sdf_sphere:
    case sdf_cut_sphere:
        return make_box3fa(prim->p0 - radius, prim->p0 + radius);
    case sdf_torus:
        const uniform float r = prim->r0 + prim->r1 + prim->userParams.x;
        return make_box3fa(prim->p0 - r, prim->p0 + r);
    default:
        const uniform vec3f minV = min(prim->p0, prim->p1);
        const uniform vec3f maxV = max(prim->p0, prim->p1);
        return make_box3fa(minV - radius, maxV + radius);
    }
}

float sdfDistance(const vec3f& p, uDataPtr_t geo_, uDataPtr_t prim_, const SDFParams& params,
                  const bool processDisplacement)
{
    const Geo_ptr geo = (const Geo_ptr)geo_;
    const Prim_ptr prim = (const Prim_ptr)prim_;

    float distance = calcDistance(*prim, p, processDisplacement);
    if (processDisplacement && prim->numNeighbours > 0)
    {
        const float r0 = max(prim->r0, prim->r1);
        for (uniform uint8 i = 0; i < prim->numNeighbours; ++i)
        {
            const uniform uint64 index = getNeighbourIdx(*geo, prim->neighboursIndex, i);
            const uniform SDFGeometry& neighbour = *getPrimitive(*geo, index);
            const float neighbourDistance = calcDistance(neighbour, p, processDisplacement);
            if (neighbourDistance < 0.f)
                continue;
            const float r1 = max(neighbour.r0, neighbour.r1);
            const float blendFactor = lerp(params.blendLerpFactor, min(r0, r1), max(r0, r1));
            distance = sminPoly(neighbourDistance, distance, blendFactor * params.blendFactor);
        }
    }
    return distance;
}

void SDFGeometries_bounds(const RTCBoundsFunctionArguments* uniform args)
{
    const Geo_ptr geo = (Geo_ptr)args->geometryUserPtr;
    const uniform uint64 idx = primToIdx(*geo, args->primID);
    const Prim_ptr prim = getPrimitive(*geo, idx);

    *((box3fa * uniform) args->bounds_o) = sdfBounds((uDataPtr_t)geo, (uDataPtr_t)prim);
}

unmasked void SDFGeometries_intersect(const RTCIntersectFunctionNArguments* uniform args)
{
    const Geo_ptr geo = (Geo_ptr)args->geometryUserPtr;
    const uniform uint64 idx = primToIdx(*geo, args->primID);
    const Prim_ptr prim = getPrimitive(*geo, idx);

    varying Ray* uniform ray = (varying Ray * uniform) args->rayhit;

    // dummy parameter struct
    SDFParams sdfParams;
    sdfParams.eye = ray->org;
    sdfParams.epsilon = geo->epsilon;
    sdfParams.nbMarchIterations = geo->nbMarchIterations;
    sdfParams.blendFactor = geo->blendFactor;
    sdfParams.blendLerpFactor = geo->blendLerpFactor;
    sdfParams.omega = geo->omega;
    sdfParams.distance = geo->distance;

    bool processDisplacement = true;
    const float t_in =
        raymarching(*ray, sdfDistance, sdfBounds, (uDataPtr_t)geo, (uDataPtr_t)prim, sdfParams, processDisplacement);

    if (t_in > 0 && t_in > ray->t0 && t_in < ray->t)
    {
        const vec3f pos = ray->org + t_in * ray->dir;

        ray->primID = args->primID;
        ray->geomID = geo->super.geomID;
        ray->instID = args->context->instID[0];
        ray->t = t_in;

        // TODO epsilon based on pixel radius
        ray->Ng = computeNormal(pos, sdfDistance, calcEpsilon(ray->org, t_in), (uDataPtr_t)geo, (uDataPtr_t)prim,
                                sdfParams, processDisplacement);
    }
}

static void SDFGeometries_postIntersect(uniform Geometry* uniform geometry, uniform Model* uniform model,
                                        varying DifferentialGeometry& dg, const varying Ray& ray, uniform int64 flags)
{
    dg.geometry = geometry;
    vec3f Ng = ray.Ng;
    vec3f Ns = Ng;

    if (flags & DG_NORMALIZE)
    {
        Ng = normalize(Ng);
        Ns = normalize(Ns);
    }
    if (flags & DG_FACEFORWARD)
    {
        if (dot(ray.dir, Ng) >= 0.f)
            Ng = neg(Ng);
        if (dot(ray.dir, Ns) >= 0.f)
            Ns = neg(Ns);
    }

    dg.Ng = Ng;
    dg.Ns = Ns;
}

export void* uniform SDFGeometries_create(void* uniform cppEquivalent)
{
    Geo_ptr geom = uniform new uniform SDFGeometries;
    Geometry_Constructor(&geom->super, cppEquivalent, SDFGeometries_postIntersect, 0, 0, 0);
    return geom;
}

export void SDFGeometriesGeometry_set(void* uniform _self, void* uniform _model, void* uniform data,
                                      int uniform numPrimitives, void* uniform neighbours, int uniform numNeighbours,
                                      void* uniform geometries, float uniform epsilon, int uniform nbMarchIterations,
                                      float uniform blendFactor, float uniform blendLerpFactor, float uniform omega,
                                      float uniform distance)
{
    Geo_ptr self = (Geo_ptr)_self;
    uniform Model* uniform model = (uniform Model * uniform) _model;

    RTCGeometry geom = rtcNewGeometry(ispc_embreeDevice(), RTC_GEOMETRY_TYPE_USER);
    uniform uint32 geomID = rtcAttachGeometry(model->embreeSceneHandle, geom);

    self->super.model = model;
    self->super.geomID = geomID;
    self->super.numPrimitives = numPrimitives;
    self->geometryRefs = (uniform uint64 * uniform) data;
    self->neighbours = (uniform uint64 * uniform) neighbours;
    self->geometries = (uniform SDFGeometry * uniform) geometries;
    self->epsilon = epsilon;
    self->nbMarchIterations = nbMarchIterations;
    self->blendFactor = blendFactor;
    self->blendLerpFactor = blendLerpFactor;
    self->omega = omega;
    self->distance = distance;

    // NOTE: self->data is always smaller than self->geometries
    self->useSafeIncrement =
        needsSafeIncrement(self->geometries, numPrimitives) || needsSafeIncrement(self->neighbours, numNeighbours);

    rtcSetGeometryUserData(geom, self);
    rtcSetGeometryUserPrimitiveCount(geom, numPrimitives);
    rtcSetGeometryBoundsFunction(geom, (uniform RTCBoundsFunction)&SDFGeometries_bounds, self);
    rtcSetGeometryIntersectFunction(geom, (uniform RTCIntersectFunctionN)&SDFGeometries_intersect);
    rtcSetGeometryOccludedFunction(geom, (uniform RTCOccludedFunctionN)&SDFGeometries_intersect);
    rtcCommitGeometry(geom);
    rtcReleaseGeometry(geom);
}
