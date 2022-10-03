/* Copyright (c) 2015-2017, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 * Author: Jafet Villafranca Diaz <jafet.villafrancadiaz@epfl.ch>
 *
 * Ray-cone intersection:
 * based on Ching-Kuang Shene (Graphics Gems 5, p. 227-230)
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

 #if 0
 #include <optix.h>

 #include <sutil/vec_math.h>
 
#define OFFSET_USER_DATA 0
#define OFFSET_CENTER (OFFSET_USER_DATA + 2)
#define OFFSET_UP (OFFSET_CENTER + 3)
#define OFFSET_CENTER_RADIUS (OFFSET_UP + 3)
#define OFFSET_UP_RADIUS (OFFSET_CENTER_RADIUS + 1)
#define OFFSET_TIMESTAMP (OFFSET_UP_RADIUS + 1)
#define OFFSET_TEX_COORDS (OFFSET_TIMESTAMP + 1)

// Global variables
rtDeclareVariable(unsigned int, cone_size, , );

rtBuffer<float> cones;

// Geometry specific variables
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(unsigned long, simulation_idx, attribute simulation_idx, );

template <bool use_robust_method>
static __device__ void intersect_cone(int primIdx)
{
    const int idx = primIdx * cone_size;

    const unsigned long userData =
        *((unsigned long*)(&cones[idx + OFFSET_USER_DATA]));

    float3 v0 = {cones[idx + OFFSET_CENTER], cones[idx + OFFSET_CENTER + 1],
                 cones[idx + OFFSET_CENTER + 2]};
    float3 v1 = {cones[idx + OFFSET_UP], cones[idx + OFFSET_UP + 1],
                 cones[idx + OFFSET_UP + 2]};
    float radius0 = cones[idx + OFFSET_CENTER_RADIUS];
    float radius1 = cones[idx + OFFSET_UP_RADIUS];

    if (radius0 < radius1)
    {
        // swap radii and positions, so radius0 and v0 are always at the bottom
        float tmpRadius = radius1;
        radius1 = radius0;
        radius0 = tmpRadius;

        float3 tmpPos = v1;
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
    float3 n = normalize(cross(ray.direction, V - ray.origin));
    const float dotNV = dot(n, v);
    if (dotNV > 0.f)
        n = n * -1.f;

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
    const float3 rayApex = V - ray.origin;

    const float3 normal1 = cross(ray.direction, delta1);
    const float length1 = length(normal1);

    if (length1 == 0.f)
        return;

    const float r1 = dot(cross(rayApex, delta1), normal1) / (length1 * length1);

    const float3 normal2 = cross(ray.direction, delta2);
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

    bool check_second = true;
    if (t_in > 0.f)
    {
        const float3 p1 = ray.origin + t_in * ray.direction;
        // consider only the parts within the extents of the truncated cone
        if (dot(p1 - v1, v) > 0.f && dot(p1 - v0, v) < 0.f)
        {
            if (rtPotentialIntersection(t_in))
            {
                const float3 surfaceVec = normalize(p1 - V);
                geometric_normal = shading_normal =
                    cross(cross(v, surfaceVec), surfaceVec);
                simulation_idx = userData;
                if (rtReportIntersection(0))
                    check_second = false;
            }
        }
    }

    if (check_second)
    {
        if (t_out > 0.f)
        {
            const float3 p2 = ray.origin + t_out * ray.direction;
            // consider only the parts within the extents of the truncated cone
            if (dot(p2 - v1, v) > 0.f && dot(p2 - v0, v) < 0.f)
            {
                if (rtPotentialIntersection(t_out))
                {
                    const float3 surfaceVec = normalize(p2 - V);
                    geometric_normal = shading_normal =
                        cross(cross(v, surfaceVec), surfaceVec);
                    simulation_idx = userData;
                    rtReportIntersection(0);
                }
            }
        }
    }
}

RT_PROGRAM void intersect(int primIdx)
{
    intersect_cone<false>(primIdx);
}

RT_PROGRAM void robust_intersect(int primIdx)
{
    intersect_cone<true>(primIdx);
}

RT_PROGRAM void bounds(int primIdx, float result[6])
{
    const int idx = primIdx * cone_size;
    const float3 v0 = {cones[idx + OFFSET_CENTER],
                       cones[idx + OFFSET_CENTER + 1],
                       cones[idx + OFFSET_CENTER + 2]};
    const float3 v1 = {cones[idx + OFFSET_UP], cones[idx + OFFSET_UP + 1],
                       cones[idx + OFFSET_UP + 2]};
    const float radius =
        max(cones[idx + OFFSET_CENTER_RADIUS], cones[idx + OFFSET_UP_RADIUS]);

    const float3 V0 = {min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z)};
    const float3 V1 = {max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z)};

    optix::Aabb* aabb = (optix::Aabb*)result;

    if (radius > 0.0f && !isinf(radius))
    {
        aabb->m_min = V0 - radius;
        aabb->m_max = V1 + radius;
    }
    else
        aabb->invalidate();
}
#else
//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "optixSphere.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}


static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                prd
        )
{
    unsigned int p0, p1, p2;
    p0 = __float_as_uint( prd->x );
    p1 = __float_as_uint( prd->y );
    p2 = __float_as_uint( prd->z );
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1, p2 );
    prd->x = __uint_as_float( p0 );
    prd->y = __uint_as_float( p1 );
    prd->z = __uint_as_float( p2 );
}


static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}


static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
            __uint_as_float( optixGetPayload_0() ),
            __uint_as_float( optixGetPayload_1() ),
            __uint_as_float( optixGetPayload_2() )
            );
}


extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    const float3      U      = rtData->camera_u;
    const float3      V      = rtData->camera_v;
    const float3      W      = rtData->camera_w;
    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    const float3 origin      = rtData->cam_eye;
    const float3 direction   = normalize( d.x * U + d.y * V + W );
    float3       payload_rgb = make_float3( 0.5f, 0.5f, 0.5f );
    trace( params.handle,
            origin,
            direction,
            0.00f,  // tmin
            1e16f,  // tmax
            &payload_rgb );

    params.image[idx.y * params.image_width + idx.x] = make_color( payload_rgb );
}


extern "C" __global__ void __miss__ms()
{
    MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3    payload = getPayload();
    setPayload( make_float3( rt_data->r, rt_data->g, rt_data->b ) );
}


extern "C" __global__ void __closesthit__ch()
{
    float  t_hit = optixGetRayTmax();
    // Backface hit not used.
    //float  t_hit2 = __uint_as_float( optixGetAttribute_0() ); 

    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();

    const unsigned int           prim_idx    = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           sbtGASIndex = optixGetSbtGASIndex();

    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, prim_idx, sbtGASIndex, 0.f, &q );

    float3 world_raypos = ray_orig + t_hit * ray_dir;
    float3 obj_raypos   = optixTransformPointFromWorldToObjectSpace( world_raypos );
    float3 obj_normal   = ( obj_raypos - make_float3( q ) ) / q.w;
    float3 world_normal = normalize( optixTransformNormalFromObjectToWorldSpace( obj_normal ) );

    setPayload( world_normal * 0.5f + 0.5f );
}
#endif