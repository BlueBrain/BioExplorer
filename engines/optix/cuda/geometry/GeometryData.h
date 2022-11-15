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
#pragma once

#include <optix.h>

#include <cuda/BufferView.h>

#include <sutil/vec_math.h>

#ifndef __CUDACC_RTC__
#include <cassert>
#else
#define assert(x) /*nop*/
#endif

namespace brayns
{
#define float3_as_uints(u) \
    __float_as_uint(u.x), __float_as_uint(u.y), __float_as_uint(u.z)

// unaligned equivalent of float2
struct Vec2f
{
    SUTIL_HOSTDEVICE operator float2() const { return {x, y}; }

    float x, y;
};

struct Vec4f
{
    SUTIL_HOSTDEVICE operator float4() const { return {x, y, z, w}; }

    float x, y, z, w;
};

struct GeometryData
{
    enum Type
    {
        TRIANGLE_MESH = 0,
        SPHERE = 1,
        CYLINDER = 2,
        CONE = 3,
        LINEAR_CURVE_ARRAY = 4,
        QUADRATIC_CURVE_ARRAY = 5,
        CUBIC_CURVE_ARRAY = 6,
        CATROM_CURVE_ARRAY = 7,
    };

    // The number of supported texture spaces per mesh.
    static const unsigned int num_texcoords = 2;

    struct TriangleMesh
    {
        GenericBufferView indices;
        BufferView<float3> positions;
        BufferView<float3> normals;
        BufferView<Vec2f> texcoords[num_texcoords]; // The buffer view may not
                                                    // be aligned, so don't use
                                                    // float2
        BufferView<Vec4f> colors; // The buffer view may not be aligned, so
                                  // don't use float4
    };

    struct Sphere
    {
        float3 center;
        float radius;
    };

    struct Cylinder
    {
        float3 center;
        float3 up;
        float radius;
    };

    struct Cone
    {
        float3 center;
        float3 up;
        float centerRadius;
        float upRadius;
    };

    struct Curves
    {
        BufferView<float2> strand_u;   // strand_u at segment start per segment
        GenericBufferView strand_i;    // strand index per segment
        BufferView<uint2> strand_info; // info.x = segment base
                                       // info.y = strand length (segments)
    };

    Type type;

    union
    {
        TriangleMesh triangle_mesh;
        Sphere sphere;
        Cylinder cylinder;
        Cone cone;
        Curves curves;
    };
};
} // namespace brayns
