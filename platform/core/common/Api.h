/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * Responsible Author: Daniel.Nachbaur@epfl.ch
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

#include <Defines.h>

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
