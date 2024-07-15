/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

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

#if __cplusplus
#include <platform/core/common/Types.h>
#define VEC3_TYPE core::Vector3f
#define UINT64_T uint64_t
#define UINT8_T uint8_t
#endif

#if ISPC
#define VEC3_TYPE vec3f
#define UINT64_T unsigned int64
#define UINT8_T unsigned int8
#endif
