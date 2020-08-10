/* Copyright (c) 2018, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#ifndef BIOEXPLORER_LOG_H
#define BIOEXPLORER_LOG_H

#include <iostream>

namespace bioexplorer
{
#define PLUGIN_ERROR std::cerr << "[ERROR] [BIO_EXPLORER] "
#define PLUGIN_WARN std::cerr << "[WARN ] [BIO_EXPLORER] "
#define PLUGIN_INFO std::cout << "[INFO ] [BIO_EXPLORER] "
#ifdef NDEBUG
#define PLUGIN_DEBUG \
    if (false)       \
    std::cout
#else
#define PLUGIN_DEBUG std::cout << "[DEBUG] [BIO_EXPLORER] "
#endif

#define PLUGIN_THROW(exc)                        \
    {                                            \
        PLUGIN_ERROR << exc.what() << std::endl; \
        throw exc;                               \
    }
} // namespace bioexplorer

#endif // BIOEXPLORER_LOG_H
