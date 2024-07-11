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

#include "GeneralSettings.h"

#include <platform/core/common/Timer.h>

#include <iostream>

namespace bioexplorer
{
namespace common
{
#define PLUGIN_PREFIX "BIO_EXPLORER    "
#define PROGRESS_BAR_SIZE 50

#define PLUGIN_ERROR(__msg) std::cerr << "E [" << PLUGIN_PREFIX << "] " << __msg << std::endl;
#define PLUGIN_WARN(__msg) std::cerr << "W [" << PLUGIN_PREFIX << "] " << __msg << std::endl;
#define PLUGIN_INFO(__level, __msg)                                                        \
    if (__level <= bioexplorer::common::GeneralSettings::getInstance()->getLoggingLevel()) \
        std::cout << "I [" << PLUGIN_PREFIX << "] " << __msg << std::endl;
#ifdef NDEBUG
#define PLUGIN_DEBUG(__msg) ;
#else
#define PLUGIN_DEBUG(__msg)                                                         \
    if (bioexplorer::common::GeneralSettings::getInstance()->getLoggingLevel() > 0) \
    {                                                                               \
        std::cout << "D [" << PLUGIN_PREFIX << "] " << __msg << std::endl;          \
    }
#endif
#define PLUGIN_TIMER(__time, __msg)                                                 \
    if (bioexplorer::common::GeneralSettings::getInstance()->getLoggingLevel() > 0) \
        std::cout << "T [" << PLUGIN_PREFIX << "] [" << __time << "] " << __msg << std::endl;

#define PLUGIN_DB_INFO(__level, __msg)                                                       \
    if (__level <= bioexplorer::common::GeneralSettings::getInstance()->getDBLoggingLevel()) \
        std::cout << "I [" << PLUGIN_PREFIX << "] [DB] " << __msg << std::endl;
#ifdef NDEBUG
#define PLUGIN_DB_DEBUG(__msg) ;
#else
#define PLUGIN_DB_DEBUG(__msg)                                                        \
    if (bioexplorer::common::GeneralSettings::getInstance()->getDBLoggingLevel() > 0) \
    {                                                                                 \
        std::cout << "D [" << PLUGIN_PREFIX << "] [DB] " << __msg << std::endl;       \
    }
#endif
#define PLUGIN_DB_TIMER(__time, __msg)                                                \
    if (bioexplorer::common::GeneralSettings::getInstance()->getDBLoggingLevel() > 0) \
        std::cout << "T [" << PLUGIN_PREFIX << "] [DB] [" << __time << "] " << __msg << std::endl;

#define PLUGIN_THROW(__msg)              \
    {                                    \
        throw std::runtime_error(__msg); \
    }

#define PLUGIN_PROGRESS(__msg, __progress, __maxValue)                                                          \
    {                                                                                                           \
        if (bioexplorer::common::GeneralSettings::getInstance()->getLoggingLevel() > 0)                         \
        {                                                                                                       \
            std::cout << "I [" << PLUGIN_PREFIX << "] [";                                                       \
            const float __mv = float(__maxValue);                                                               \
            const float __p = float(__progress + 1);                                                            \
            const uint32_t __pos = std::min(PROGRESS_BAR_SIZE, int(__p / __mv * PROGRESS_BAR_SIZE));            \
            for (uint32_t __i = 0; __i < PROGRESS_BAR_SIZE; ++__i)                                              \
            {                                                                                                   \
                if (__i < __pos)                                                                                \
                    std::cout << "=";                                                                           \
                else if (__i == __pos)                                                                          \
                    std::cout << ">";                                                                           \
                else                                                                                            \
                    std::cout << " ";                                                                           \
            }                                                                                                   \
            std::cout << "] " << std::min(__pos * 2, uint32_t(PROGRESS_BAR_SIZE * 2)) << "% " << __msg << "\r"; \
            std::cout.flush();                                                                                  \
        }                                                                                                       \
    }

} // namespace common
} // namespace bioexplorer
