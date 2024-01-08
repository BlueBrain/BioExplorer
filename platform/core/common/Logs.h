/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
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

#include <iostream>

namespace core
{
#define CORE_PREFIX "CORE            " // 16 characters

#define PROGRESS_BAR_SIZE 50

#define CORE_ERROR(__msg) std::cerr << "E [" << CORE_PREFIX << "] " << __msg << std::endl;
#define CORE_WARN(__msg) std::cerr << "W [" << CORE_PREFIX << "] " << __msg << std::endl;
#define CORE_INFO(__msg) std::cout << "I [" << CORE_PREFIX << "] " << __msg << std::endl;

#ifdef NDEBUG
#define CORE_DEBUG(__msg)
#else
#define CORE_DEBUG(__msg) std::cout << "D [" << CORE_PREFIX << "] " << __msg << std::endl;
#endif
#define CORE_TIMER(__time, __msg) std::cout << "T [" << CORE_PREFIX << "] [" << __time << "] " << __msg << std::endl;

#define CORE_THROW(__msg)                \
    {                                    \
        throw std::runtime_error(__msg); \
    }

#define CORE_PROGRESS(__msg, __progress, __maxValue)                                                        \
    {                                                                                                       \
        std::cout << "I [" << CORE_PREFIX << "] [";                                                         \
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
    }

#define PLUGIN_REGISTER_ENDPOINT(__msg) CORE_INFO("Registering end-point '" << __msg << "'");
#define PLUGIN_REGISTER_RENDERER(__msg) CORE_INFO("Registering renderer '" << __msg << "'");
#define PLUGIN_REGISTER_LOADER(__msg) CORE_INFO("Registering loader '" << __msg << "'");
#define PLUGIN_REGISTER_CAMERA(__msg) CORE_INFO("Registering camera '" << __msg << "'");
} // namespace core