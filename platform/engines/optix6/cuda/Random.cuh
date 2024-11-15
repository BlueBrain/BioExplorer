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

#pragma once

#include <optixu/optixu_math_namespace.h>

template <uint N>
static __host__ __device__ __inline__ uint tea(uint val0, uint val1)
{
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;

    for (uint n = 0; n < N; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

// Generate random uint in [0, 2^24)
static __host__ __device__ __inline__ uint lcg(uint &prev)
{
    const uint LCG_A = 1664525u;
    const uint LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

static __host__ __device__ __inline__ uint lcg2(uint &prev)
{
    prev = (prev * 8121 + 28411) % 134456;
    return prev;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(uint &prev)
{
    return ((float)lcg(prev) / (float)0x01000000);
}

// Multiply with carry
static __host__ __inline__ uint mwc()
{
    static unsigned long long r[4];
    static unsigned long long carry;
    static bool init = false;
    if (!init)
    {
        init = true;
        uint seed = 7654321u, seed0, seed1, seed2, seed3;
        r[0] = seed0 = lcg2(seed);
        r[1] = seed1 = lcg2(seed0);
        r[2] = seed2 = lcg2(seed1);
        r[3] = seed3 = lcg2(seed2);
        carry = lcg2(seed3);
    }

    unsigned long long sum = 2111111111ull * r[3] + 1492ull * r[2] + 1776ull * r[1] + 5115ull * r[0] + 1ull * carry;
    r[3] = r[2];
    r[2] = r[1];
    r[1] = r[0];
    r[0] = static_cast<uint>(sum);        // lower half
    carry = static_cast<uint>(sum >> 32); // upper half
    return static_cast<uint>(r[0]);
}

static __host__ __inline__ uint random1u()
{
#if 0
  return rand();
#else
    return mwc();
#endif
}

static __host__ __inline__ optix::uint2 random2u()
{
    return optix::make_uint2(random1u(), random1u());
}

static __host__ __inline__ void fillRandBuffer(uint *seeds, uint N)
{
    for (uint i = 0; i < N; ++i)
        seeds[i] = mwc();
}

static __host__ __device__ __inline__ uint rot_seed(uint seed, uint frame)
{
    return seed ^ frame;
}
