/*
 * Copyright (c) 2020-2023, EPFL/Blue Brain Project
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