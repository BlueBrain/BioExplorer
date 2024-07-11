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

#include <platform/core/common/PropertyMap.h>

using namespace core;

namespace spaceexplorer
{
namespace blackhole
{
static constexpr int BLACK_HOLE_DEFAULT_RENDERER_NB_DISKS = 20;
static constexpr bool BLACK_HOLE_DEFAULT_RENDERER_DISPLAY_GRID = false;
static constexpr double BLACK_HOLE_DEFAULT_RENDERER_DISK_ROTATION_SPEED = 3.0;
static constexpr int BLACK_HOLE_DEFAULT_RENDERER_TEXTURE_LAYERS = 12;
static constexpr double BLACK_HOLE_DEFAULT_RENDERER_SIZE = 12;
static constexpr double MAX_BLACK_HOLE_SIZE = 100.0;

static const Property BLACK_HOLE_RENDERER_PROPERTY_NB_DISKS = {
    "nbDisks", BLACK_HOLE_DEFAULT_RENDERER_NB_DISKS, 2, 128, {"Number of disks"}};
static const Property BLACK_HOLE_RENDERER_PROPERTY_DISPLAY_GRID = {"displayGrid",
                                                                   BLACK_HOLE_DEFAULT_RENDERER_DISPLAY_GRID,
                                                                   {"Display grid"}};
static const Property BLACK_HOLE_RENDERER_PROPERTY_DISK_ROTATION_SPEED = {
    "diskRotationSpeed", BLACK_HOLE_DEFAULT_RENDERER_DISK_ROTATION_SPEED, 1., 10., {"Disk rotation speed"}};
static const Property BLACK_HOLE_RENDERER_PROPERTY_DISK_TEXTURE_LAYERS = {
    "diskTextureLayers", BLACK_HOLE_DEFAULT_RENDERER_TEXTURE_LAYERS, 2, 100, {"Disk texture layers"}};
static const Property BLACK_HOLE_RENDERER_PROPERTY_SIZE = {
    "size", BLACK_HOLE_DEFAULT_RENDERER_SIZE, 0.1, MAX_BLACK_HOLE_SIZE, {"Black hole size"}};

} // namespace blackhole
} // namespace spaceexplorer