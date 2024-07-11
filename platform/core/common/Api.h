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

#if defined(_MSC_VER) || defined(__declspec)
#define CORE_DLLEXPORT __declspec(dllexport)
#define CORE_DLLIMPORT __declspec(dllimport)
#else // _MSC_VER
#define CORE_DLLEXPORT
#define CORE_DLLIMPORT
#endif // _MSC_VER

#if defined(CORE_STATIC)
#define PLATFORM_API
#elif defined(CORE_SHARED)
#define PLATFORM_API CORE_DLLEXPORT
#else
#define PLATFORM_API CORE_DLLIMPORT
#endif
