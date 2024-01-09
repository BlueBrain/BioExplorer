/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
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

#include "Field.ih"
#include "utils/SafeIncrement.ih"

#include <ospray/SDK/common/Model.ih>
#include <ospray/SDK/geometry/Geometry.ih>

DEFINE_SAFE_INCREMENT(Field);

float treeWalker(const uniform Field* uniform self, const vec3f& point, const varying float distance,
                 const varying float cutoff, const varying int32 index = 0)
{
    const uint64 begin = self->indices[index * 2];
    const uint64 end = self->indices[index * 2 + 1];
    if (begin == 0 && end == 0)
        return self->values[index] / (distance * distance);

    float voxelValue = 0.f;
    for (uint64 childIndex = begin; childIndex <= end; ++childIndex)
    {
        const uint64 idx = childIndex * 4; // 4 Floats per value (position + value)
        const vec3f childPosition = make_vec3f(self->values[idx], self->values[idx + 1], self->values[idx + 2]);
        const float d = length(point - childPosition);
        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate events in the child node, we take the
            // precomputed value of node instead
            voxelValue += self->values[childIndex + 3] / (d * d);
        }
        else
            // Dive into the child node and compute its contents
            voxelValue += treeWalker(self, point, d, cutoff / 2.f, childIndex);
    }
    return voxelValue;
}

vec3f treeWalker3(const uniform Field* uniform self, const vec3f& point, const varying float distance,
                  const varying float cutoff, const varying int32 index = 0)
{
    const uint64 begin = self->indices[index * 2];
    const uint64 end = self->indices[index * 2 + 1];
    const uint64 idx = index * 6; // 6 Floats per value (position + direction)

    if (begin == 0 && end == 0)
        return make_vec3f(self->values[idx], self->values[idx + 1], self->values[idx + 2]) /
               (self->distance * self->distance);

    vec3f voxelValue = make_vec3f(0.f);
    for (uint64 childIndex = begin; childIndex <= end; ++childIndex)
    {
        uint64 idx = childIndex * 3;
        const vec3f childPosition = make_vec3f(self->values[idx], self->values[idx + 1], self->values[idx + 2]);
        const float d = length(point - childPosition);
        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate events in the child node, we take the
            // precomputed value of node instead
            const vec3f value = make_vec3f(self->values[idx + 3], self->values[idx + 4], self->values[idx + 5]);
            voxelValue = voxelValue + value / (d * d);
        }
        else
            // Dive into the child node and compute its contents
            voxelValue = voxelValue + treeWalker3(self, point, d, cutoff / 2.f, childIndex);
    }
    return voxelValue;
}

float get_voxel_value(const uniform Field* uniform self, const vec3f& p)
{
    switch (self->dataType)
    {
    case point:
        return treeWalker(self, p, self->distance, self->cutoff, 0u);
    case vector:
        const vec3f sampleValue = treeWalker3(self, p, self->distance, self->cutoff, 0u);
        return clamp(length(sampleValue), self->transferFunction->valueRange.x, self->transferFunction->valueRange.y);
    }
}

unmasked void Field_bounds(const RTCBoundsFunctionArguments* uniform args)
{
    const uniform Field* uniform self = (uniform Field * uniform) args->geometryUserPtr;
    const uniform int primID = args->primID;

    box3fa* uniform bbox = (box3fa * uniform) args->bounds_o;
    *bbox = make_box3fa(self->super.gridOrigin,
                        self->super.gridOrigin + make_vec3f(self->super.dimensions) * self->super.gridSpacing);
}

unmasked void Field_intersect(const RTCIntersectFunctionNArguments* uniform args)
{
    const uniform Field* uniform self = (uniform Field * uniform) args->geometryUserPtr;
    const uniform int primID = args->primID;

    varying Ray* uniform ray = (varying Ray * uniform) args->rayhit;

    const vec3f boxMin = self->super.gridOrigin;
    const vec3f boxMax = self->super.gridOrigin + make_vec3f(self->super.dimensions) * self->super.gridSpacing;

    const vec3f a = (boxMin - ray->org) / ray->dir;
    const vec3f b = (boxMax - ray->org) / ray->dir;
    const vec3f near = min(a, b);
    const vec3f far = max(a, b);
    float t0 = reduce_max(near);
    float t1 = reduce_min(far);

    const float diag = max(self->super.gridSpacing.x, max(self->super.gridSpacing.y, self->super.gridSpacing.z));
    const float step = max(0.1f, diag / self->samplingRate);
    const float random = 0; // TODO: rnd(seed) * step;

    // Apply ray clipping
    t0 = max(t0, ray->t0);
    t1 = min(t1, ray->t);

    if (t0 > 0.f && t0 <= t1)
    {
        float t = t0 + random;
        while (t < t1)
        {
            const vec3f p = ray->org + t * ray->dir;
            const vec3f p0 = (p - self->super.gridOrigin) / self->super.gridSpacing;
            const float voxelValue = get_voxel_value(self, p0);
            const float voxelAlpha = self->transferFunction->getOpacityForValue(self->transferFunction, voxelValue);
            const vec3f voxelColor = self->transferFunction->getColorForValue(self->transferFunction, voxelValue);
            if (voxelAlpha > 0.f)
            {
                vec3f normal = make_vec3f(voxelValue);
                if (self->gradientShadingEnabled)
                {
                    normal = make_vec3f(0);
                    const vec3f positions[6] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
                    for (uint64 i = 0; i < 6; ++i)
                    {
                        const vec3f p1 = p0 + (positions[i] * self->gradientOffset);
                        const float voxelValue = get_voxel_value(self, p1);
                        normal = normal + voxelValue * positions[i];
                    }
                    normal = normalize(-1.f * normal);
                }
                ray->primID = primID;
                // ray->geomID = self->super.geomID;
                ray->instID = args->context->instID[0];
                ray->t = t;
                ray->Ng = normal;
                break;
            }
            t += step;
        }
    }
}

static void Field_postIntersect(uniform Geometry* uniform geometry, uniform Model* uniform model,
                                varying DifferentialGeometry& dg, const varying Ray& ray, uniform int64 flags)
{
    uniform Field* uniform this = (uniform Field * uniform) geometry;
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

void Field_Constructor(Field* uniform volume,
                       /*! pointer to the c++-equivalent class of this entity */
                       void* uniform cppEquivalent, const uniform int voxelType, const uniform vec3i& dimensions)
{
    StructuredVolume_Constructor(&volume->super, cppEquivalent, dimensions);
}

export void* uniform Field_createInstance(void* uniform cppEquivalent, const uniform int voxelType,
                                          const uniform vec3i& dimensions)
{
    // The volume container.
    Field* uniform volume = uniform new uniform Field;
    Field_Constructor(volume, cppEquivalent, voxelType, dimensions);

    return volume;
}

#if 0
export void* uniform Field_create(void* uniform cppEquivalent)
{
    uniform Field* uniform geom = uniform new uniform Field;
    Geometry_Constructor(&geom->super.super, cppEquivalent, Field_postIntersect, NULL, NULL, 0, NULL);
    return geom;
}
#endif

export void Field_set(void* uniform _self, const uniform vec3i& dimensions, const uniform vec3f& spacing,
                      const uniform vec3f& gridOrigin, void* uniform indices, void* uniform values)
{
    uniform Field* uniform self = (uniform Field * uniform) _self;
    RTCGeometry geom = rtcNewGeometry(ispc_embreeDevice(), RTC_GEOMETRY_TYPE_USER);
    // uniform uint32 geomID = rtcAttachGeometry(model->embreeSceneHandle, geom);
    // self->super.geomID = geomID;

    // uniform Model* uniform model = (uniform Model * uniform) _model;
    // self->super.model = model;

    // self->super.snumPrimitives = 1;
    self->super.dimensions = dimensions;
    self->super.gridOrigin = gridOrigin;
    self->super.gridSpacing = spacing;
    self->indices = (uniform uint64 * uniform) indices;
    self->values = (uniform float* uniform)values;

    rtcSetGeometryUserData(geom, self);
    rtcSetGeometryUserPrimitiveCount(geom, 1);
    rtcSetGeometryBoundsFunction(geom, (uniform RTCBoundsFunction)&Field_bounds, self);
    rtcSetGeometryIntersectFunction(geom, (uniform RTCIntersectFunctionN)&Field_intersect);
    rtcSetGeometryOccludedFunction(geom, (uniform RTCOccludedFunctionN)&Field_intersect);
    rtcCommitGeometry(geom);
    rtcReleaseGeometry(geom);
}
