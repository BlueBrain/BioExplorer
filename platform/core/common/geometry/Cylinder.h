/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include "CommonDefines.h"

#include <Defines.h>

#if __cplusplus
namespace core
{
#endif
struct Cylinder
{
#if __cplusplus
    Cylinder(const Vector3f c = {0.f, 0.f, 0.f}, const Vector3f u = {0.f, 0.f, 0.f}, const float r = 0.f,
             const uint64_t data = 0)
        : userData(data)
        , center(c)
        , up(u)
        , radius(r)
    {
    }
#endif

    UINT64_T userData;
    VEC3_TYPE center;
    VEC3_TYPE up;
    float radius;
    __MEMORY_ALIGNMENT__
};

#if __cplusplus
} // Platform
#endif
