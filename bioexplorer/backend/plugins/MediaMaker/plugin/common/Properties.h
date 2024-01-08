/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
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

#include <platform/core/common/PropertyMap.h>

namespace bioexplorer
{
namespace mediamaker
{
static const std::string PLUGIN_API_PREFIX = "mm-";

static const char* RENDERER_ALBEDO = "albedo";
static const char* RENDERER_AMBIENT_OCCLUSION = "ambient_occlusion";
static const char* RENDERER_DEPTH = "depth";
static const char* RENDERER_SHADOW = "shadow";
static const char* RENDERER_SHADING_NORMAL = "raycast_Ns";
static const char* RENDERER_GEOMETRY_NORMAL = "raycast_Ng";
static const char* RENDERER_RADIANCE = "radiance";

static constexpr double DEFAULT_MEDIA_MAKER_RENDERER_DEPTH_INFINITY = 1.6;
static const core::Property MEDIA_MAKER_RENDERER_PROPERTY_DEPTH_INFINITY = {"infinity", 1e6, 0., 1e6, {"Infinity"}};
} // namespace mediamaker
} // namespace bioexplorer