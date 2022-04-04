/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "GeneralSettings.h"

#include <iostream>

namespace bioexplorer
{
namespace common
{
#define PLUGIN_PREFIX "BE"

#define PLUGIN_ERROR(message) \
    std::cerr << "E [" << PLUGIN_PREFIX << "] " << message << std::endl;
#define PLUGIN_WARN(message) \
    std::cerr << "W [" << PLUGIN_PREFIX << "] " << message << std::endl;
#define PLUGIN_INFO(level, message)                                  \
    if (level <= bioexplorer::common::GeneralSettings::getInstance() \
                     ->getLoggingLevel())                            \
        std::cout << "I [" << PLUGIN_PREFIX << "] " << message << std::endl;
#ifdef NDEBUG
#define PLUGIN_DEBUG(message) ;
#else
#define PLUGIN_DEBUG(message) \
    std::cout << "D [" << PLUGIN_PREFIX << "] " << message << std::endl;
#endif
#define PLUGIN_TIMER(__time, __msg)                                         \
    std::cout << "T [" << PLUGIN_PREFIX << "] [" << __time << "] " << __msg \
              << std::endl;

#define PLUGIN_THROW(message)              \
    {                                      \
        throw std::runtime_error(message); \
    }

#define PLUGIN_PROGRESS(__msg, __progress, __maxValue)                  \
    {                                                                   \
        std::cout << "I [" << PLUGIN_PREFIX << "] [";                   \
        const float __mv = float(__maxValue);                           \
        const uint32_t __pos = 1 + __progress / __mv * 50.f;            \
        for (uint32_t __i = 0; __i < 50; ++__i)                         \
        {                                                               \
            if (__i < __pos)                                            \
                std::cout << "=";                                       \
            else if (__i == __pos)                                      \
                std::cout << ">";                                       \
            else                                                        \
                std::cout << " ";                                       \
        }                                                               \
        std::cout << "] " << std::min(__pos * 2, uint32_t(100)) << "% " \
                  << __msg << "\r";                                     \
        std::cout.flush();                                              \
    }

} // namespace common
} // namespace bioexplorer
