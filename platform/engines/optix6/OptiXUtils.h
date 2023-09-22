/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include <optix.h>

#include <platform/core/common/Types.h>

#include <iomanip>
#include <iostream>

namespace core
{
#define CORE_OPTIX_SAMPLE_NAME "braynsOptix7Engine"

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
} // namespace core
