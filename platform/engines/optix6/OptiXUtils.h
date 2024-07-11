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

#include <optix.h>

#include <platform/core/common/Types.h>

#include <iomanip>
#include <iostream>

namespace core
{
namespace engine
{
namespace optix
{
#define RT_DESTROY(__object)         \
    {                                \
        try                          \
        {                            \
            if (__object)            \
                __object->destroy(); \
        }                            \
        catch (...)                  \
        {                            \
        }                            \
        __object = nullptr;          \
    }

#define RT_DESTROY_MAP(__map)       \
    {                               \
        for (auto obj : __map)      \
        {                           \
            RT_DESTROY(obj.second); \
            obj.second = nullptr;   \
        }                           \
        __map.clear();              \
    }

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

void toOptiXProperties(const PropertyMap& object);
} // namespace optix
} // namespace engine
} // namespace core