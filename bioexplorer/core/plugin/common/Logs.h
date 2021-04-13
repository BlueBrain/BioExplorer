/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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
#include <thread>

namespace bioexplorer
{
#define PLUGIN_ERROR(message)                      \
    std::cerr << "[" << std::this_thread::get_id() \
              << "] [ERROR] [BIO_EXPLORER] " << message << std::endl;
#define PLUGIN_WARN(message)                       \
    std::cerr << "[" << std::this_thread::get_id() \
              << "] [WARN ] [BIO_EXPLORER] " << message << std::endl;
#define PLUGIN_INFO(message)                                              \
    if (GeneralSettings::getInstance()->getLoggingEnabled())              \
    {                                                                     \
        std::cout << "[" << std::this_thread::get_id()                    \
                  << "] [INFO ] [BIO_EXPLORER] " << message << std::endl; \
    }
#ifdef NDEBUG
#define PLUGIN_DEBUG(message)
#else
#define PLUGIN_DEBUG(message)                      \
    std::cout << "[" << std::this_thread::get_id() \
              << "] [DEBUG] [BIO_EXPLORER] " << message << std::endl;
#endif

#define PLUGIN_THROW(message)              \
    {                                      \
        PLUGIN_ERROR(message);             \
        throw std::runtime_error(message); \
    }
} // namespace bioexplorer
