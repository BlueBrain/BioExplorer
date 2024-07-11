/*
    Copyright 2019 - 0211 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <optixu/optixu_vector_types.h>

const size_t BASIC_LIGHT_TYPE_POINT = 0;
const size_t BASIC_LIGHT_TYPE_DIRECTIONAL = 1;

const size_t OPTIX_STACK_SIZE = 1200;
const size_t OPTIX_RAY_TYPE_COUNT = 2;
const size_t OPTIX_ENTRY_POINT_COUNT = 1;

const size_t MAX_TEXTURE_SIZE = 16384;

struct BasicLight
{
    union
    {
        ::optix::float3 pos;
        ::optix::float3 dir;
    };
    ::optix::float3 color;
    int casts_shadow;
    int type;
};

struct PerRayData_radiance
{
    ::optix::float4 result;
    float importance;
    int depth;
    float zDepth;
    ::optix::float3 rayDdx;
    ::optix::float3 rayDdy;
};

struct PerRayData_shadow
{
    ::optix::float3 attenuation;
};