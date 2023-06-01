/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

#include <iostream>

namespace bioexplorer
{
namespace mediamaker
{
#define PLUGIN_PREFIX "OD"

#define PLUGIN_ERROR(message) std::cerr << "E [" << PLUGIN_PREFIX << "] " << message << std::endl;
#define PLUGIN_WARN(message) std::cerr << "W [" << PLUGIN_PREFIX << "] " << message << std::endl;
#define PLUGIN_INFO(message) std::cout << "I [" << PLUGIN_PREFIX << "] " << message << std::endl;
#define PLUGIN_REGISTER_ENDPOINT(__msg)         \
    std::cerr << "I [" << PLUGIN_PREFIX << "] " \
              << "Registering end-point '" << __msg << "'" << std::endl;
#define PLUGIN_REGISTER_RENDERER(__msg)         \
    std::cerr << "I [" << PLUGIN_PREFIX << "] " \
              << "Registering renderer '" << __msg << "'" << std::endl;
#define PLUGIN_REGISTER_LOADER(__msg)           \
    std::cerr << "I [" << PLUGIN_PREFIX << "] " \
              << "Registering loader '" << __msg << "'" << std::endl;
#define PLUGIN_REGISTER_CAMERA(__msg)           \
    std::cerr << "I [" << PLUGIN_PREFIX << "] " \
              << "Registering camera '" << __msg << "'" << std::endl;
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
} // namespace mediamaker
} // namespace bioexplorer
