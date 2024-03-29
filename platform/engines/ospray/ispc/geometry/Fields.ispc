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

#include "Fields.ih"
#include "utils/SafeIncrement.ih"

#include <platform/core/common/CommonTypes.h>

#include <ospray/SDK/common/Model.ih>
#include <ospray/SDK/geometry/Geometry.ih>

#define MAX_RECURSION_DEPTH 30

unmasked float treeWalker(const uniform Fields* uniform self, const varying int32 depth, const vec3f& point,
                          const varying float distance, const varying float cutoff, const varying int32 index = 0)
{
    if (depth >= MAX_RECURSION_DEPTH)
        return 0.f;

    const uint32 begin = self->indices[index * 2];
    const uint32 end = self->indices[index * 2 + 1];
    const uint32 idxData = index * FIELD_POINT_DATA_SIZE;

    if (begin == 0 && end == 0)
        return self->values[idxData + FIELD_POINT_OFFSET_VALUE] / (distance * distance);

    float voxelValue = 0.f;
    for (uint32 childIndex = begin; childIndex <= end; ++childIndex)
    {
        const uint32 childIdxData = childIndex * FIELD_POINT_DATA_SIZE;
        const vec3f childPosition = make_vec3f(self->values[childIdxData + FIELD_POINT_OFFSET_POSITION_X],
                                               self->values[childIdxData + FIELD_POINT_OFFSET_POSITION_Y],
                                               self->values[childIdxData + FIELD_POINT_OFFSET_POSITION_Z]);
        const float d = length(point - childPosition);
        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate events in the child node, we take the
            // precomputed value of node instead
            voxelValue += self->values[childIdxData + FIELD_POINT_OFFSET_VALUE] / (d * d);
        }
        else
            // Dive into the child node and compute its contents
            voxelValue += treeWalker(self, depth + 1, point, d, cutoff / 2.f, childIndex);
    }
    return voxelValue;
}

unmasked vec3f treeWalker3(const uniform Fields* uniform self, const varying int32 depth, const vec3f& point,
                           const varying float distance, const varying float cutoff, const varying int32 index = 0)
{
    if (depth >= MAX_RECURSION_DEPTH)
        return make_vec3f(0.f);

    const uint32 begin = self->indices[index * 2];
    const uint32 end = self->indices[index * 2 + 1];
    const uint32 idx = index * FIELD_VECTOR_DATA_SIZE;

    if (begin == 0 && end == 0)
        return make_vec3f(self->values[idx + FIELD_VECTOR_OFFSET_DIRECTION_X],
                          self->values[idx + FIELD_VECTOR_OFFSET_DIRECTION_Y],
                          self->values[idx + FIELD_VECTOR_OFFSET_DIRECTION_Z]) /
               (distance * distance);

    vec3f voxelValue = make_vec3f(0.f);
    for (uint32 childIndex = begin; childIndex <= end; ++childIndex)
    {
        uint32 idx = childIndex * FIELD_VECTOR_DATA_SIZE;
        const vec3f childPosition = make_vec3f(self->values[idx + FIELD_VECTOR_OFFSET_POSITION_X],
                                               self->values[idx + FIELD_VECTOR_OFFSET_POSITION_Y],
                                               self->values[idx + FIELD_VECTOR_OFFSET_POSITION_Z]);
        const float d = length(point - childPosition);
        if (d >= cutoff)
        {
            // Child is further than the cutoff distance, no need to evaluate events in the child node, we take the
            // precomputed value of node instead
            const vec3f vectorDirection = make_vec3f(self->values[idx + FIELD_VECTOR_OFFSET_DIRECTION_X],
                                                     self->values[idx + FIELD_VECTOR_OFFSET_DIRECTION_Y],
                                                     self->values[idx + FIELD_VECTOR_OFFSET_DIRECTION_Z]);
            voxelValue = voxelValue + vectorDirection / (d * d);
        }
        else
            // Dive into the child node and compute its contents
            voxelValue = voxelValue + treeWalker3(self, depth + 1, point, d, cutoff / 2.f, childIndex);
    }
    return voxelValue;
}

unmasked vec4f get_voxel_value(const uniform Fields* uniform self, const vec3f& point)
{
    switch (self->dataType)
    {
    case odt_points:
        return make_vec4f(0.f, 0.f, 0.f, treeWalker(self, 0, point, self->distance, self->cutoff));
    case odt_vectors:
        const vec3f sampleValue = treeWalker3(self, 0, point, self->distance, self->cutoff);
        return make_vec4f(normalize(sampleValue), length(sampleValue));
    }
    return make_vec4f(0.f);
}

unmasked void Field_bounds(const RTCBoundsFunctionArguments* uniform args)
{
    const uniform Fields* uniform self = (uniform Fields * uniform) args->geometryUserPtr;
    const uniform int primID = args->primID;

    box3fa* uniform bbox = (box3fa * uniform) args->bounds_o;
    *bbox = make_box3fa(self->offset, self->offset + make_vec3f(self->dimensions) * self->spacing);
}

unmasked void Field_intersect(const RTCIntersectFunctionNArguments* uniform args)
{
    const uniform Fields* uniform self = (uniform Fields * uniform) args->geometryUserPtr;
    const uniform int primID = args->primID;

    varying Ray* uniform ray = (varying Ray * uniform) args->rayhit;

    if (ray->flags == RAY_FLAG_SECONDARY)
        return;

    const vec3f boxMin = self->offset;
    const vec3f boxMax = self->offset + make_vec3f(self->dimensions) * self->spacing;

    const vec3f a = (boxMin - ray->org) / ray->dir;
    const vec3f b = (boxMax - ray->org) / ray->dir;
    const vec3f near = min(a, b);
    const vec3f far = max(a, b);
    float t0 = reduce_max(near);
    float t1 = reduce_min(far);

    const float diag = reduce_max(self->spacing);
    const float step = max(self->epsilon, diag / self->samplingRate);

    float random = 0.f;
    if (self->accumulationSteps > 0)
        random = (float)(self->accumulationCount % self->accumulationSteps) / (float)(self->accumulationSteps) * step;
    else
    {
        rdrand(&random);
        random *= step;
    }

    // Apply ray clipping
    t0 = max(t0, ray->t0);
    t1 = min(t1, ray->t);

    if (t0 <= t1)
    {
        float t = t0 + random;
        while (t < t1)
        {
            const vec3f p = ray->org + t * ray->dir;
            const vec3f p0 = (p - self->offset) / self->spacing;
            const vec4f voxelValue = get_voxel_value(self, p0);
            const float voxelAlpha = self->transferFunction->getOpacityForValue(self->transferFunction, voxelValue.w);
            if (voxelAlpha > 0.f)
            {
                vec3f normal = make_vec3f(0.f, 0.f, 1.f);
                if (self->gradientShadingEnabled)
                {
                    normal = make_vec3f(0);
                    const vec3f positions[6] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
                    for (uint32 i = 0; i < 6; ++i)
                    {
                        const vec3f p1 = p0 + (positions[i] * self->gradientOffset);
                        const vec4f voxelValue = get_voxel_value(self, p1);
                        normal = normal + voxelValue.w * positions[i];
                    }
                    normal = normalize(-1.f * normal);
                }
                ray->primID = primID;
                ray->geomID = self->super.geomID;
                ray->instID = args->context->instID[0];
                ray->t = t;
                ray->Ng = voxelValue.w * normalize(normal);
                return;
            }
            t += step;
        }
    }
    return;
}

static void Field_postIntersect(uniform Geometry* uniform geometry, uniform Model* uniform model,
                                varying DifferentialGeometry& dg, const varying Ray& ray, uniform int64 flags)
{
    uniform Fields* uniform this = (uniform Fields * uniform) geometry;
    dg.geometry = geometry;
    const vec3f nNg = normalize(ray.Ng);
    vec3f Ng = nNg;
    vec3f Ns = nNg;
    const float voxelValue = length(ray.Ng);

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
    const float opacity = this->transferFunction->getOpacityForValue(this->transferFunction, voxelValue);
    const vec3f color = this->transferFunction->getColorForValue(this->transferFunction, voxelValue);
    dg.color = make_vec4f(color, opacity);

    const float diag = reduce_max(this->spacing);
    const float step = max(this->epsilon, diag / this->samplingRate);
    dg.epsilon = step;
}

export void* uniform Field_create(void* uniform cppEquivalent)
{
    uniform Fields* uniform geom = uniform new uniform Fields;
    Geometry_Constructor(&geom->super, cppEquivalent, Field_postIntersect, NULL, NULL, 0, NULL);
    return geom;
}

export void Field_set(void* uniform _self, void* uniform _model, const uniform vec3i& dimensions,
                      const uniform vec3f& spacing, const uniform vec3f& offset, void* uniform indices,
                      void* uniform values, int uniform dataType, int uniform numPrimitives)
{
    uniform Fields* uniform self = (uniform Fields * uniform) _self;
    uniform Model* uniform model = (uniform Model * uniform) _model;

    RTCGeometry geom = rtcNewGeometry(ispc_embreeDevice(), RTC_GEOMETRY_TYPE_USER);
    uniform uint32 geomID = rtcAttachGeometry(model->embreeSceneHandle, geom);

    self->super.model = model;
    self->super.geomID = geomID;
    self->super.numPrimitives = numPrimitives;

    self->dimensions = dimensions;
    self->spacing = spacing;
    self->offset = offset;

    self->indices = (uniform uint32 * uniform) indices;
    self->values = (uniform float* uniform)values;
    self->dataType = dataType;

    rtcSetGeometryUserData(geom, self);
    rtcSetGeometryUserPrimitiveCount(geom, numPrimitives);
    rtcSetGeometryBoundsFunction(geom, (uniform RTCBoundsFunction)&Field_bounds, self);
    rtcSetGeometryIntersectFunction(geom, (uniform RTCIntersectFunctionN)&Field_intersect);
    rtcSetGeometryOccludedFunction(geom, (uniform RTCOccludedFunctionN)&Field_intersect);
    rtcCommitGeometry(geom);
    rtcReleaseGeometry(geom);
}

export void Field_commit(void* uniform _self, const uniform float& distance, const uniform float& cutoff,
                         const uniform float& gradientOffset, const uniform bool& gradientShadingEnabled,
                         const uniform bool& useOctree, const uniform float& samplingRate, const uniform float& epsilon,
                         const uniform int accumulationSteps, const uniform int accumulationCount)
{
    uniform Fields* uniform self = (uniform Fields * uniform) _self;
    self->distance = distance;
    self->cutoff = cutoff;
    self->gradientOffset = gradientOffset;
    self->gradientShadingEnabled = gradientShadingEnabled;
    self->useOctree = useOctree;
    self->samplingRate = samplingRate;
    self->epsilon = epsilon;
    self->accumulationSteps = accumulationSteps;
    self->accumulationCount = accumulationCount;
}
