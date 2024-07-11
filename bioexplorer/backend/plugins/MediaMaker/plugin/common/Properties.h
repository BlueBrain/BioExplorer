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