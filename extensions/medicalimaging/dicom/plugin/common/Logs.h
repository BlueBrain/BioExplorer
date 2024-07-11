/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include <iostream>

namespace medicalimagingexplorer
{
namespace dicom
{
#define PLUGIN_PREFIX "DICOM           "

#define PLUGIN_ERROR(message) std::cerr << "E [" << PLUGIN_PREFIX << "] " << message << std::endl;
#define PLUGIN_WARN(message) std::cerr << "W [" << PLUGIN_PREFIX << "] " << message << std::endl;
#define PLUGIN_INFO(message) std::cout << "I [" << PLUGIN_PREFIX << "] " << message << std::endl;
#ifdef NDEBUG
#define PLUGIN_DEBUG(message) ;
#else
#define PLUGIN_DEBUG(message) std::cout << "D [" << PLUGIN_PREFIX << "] " << message << std::endl;
#endif
#define PLUGIN_TIMER(__time, __msg) \
    std::cout << "T [" << PLUGIN_PREFIX << "] [" << __time << "] " << __msg << std::endl;

#define PLUGIN_THROW(message)              \
    {                                      \
        PLUGIN_ERROR(message);             \
        throw std::runtime_error(message); \
    }
} // namespace dicom
} // namespace medicalimagingexplorer
