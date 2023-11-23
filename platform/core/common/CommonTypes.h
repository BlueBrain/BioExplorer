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

/** User data */
#define NO_USER_DATA -1

enum CameraStereoMode
{
    mono = 0,
    left = 1,
    right = 2,
    side_by_side = 3
};

enum MaterialClippingMode
{
    no_clipping = 0,
    plane = 1,
    sphere = 2
};

enum MaterialShadingMode
{
    undefined_shading_mode = 0,
    basic = 1,
    diffuse = 2,
    electron = 3,
    cartoon = 4,
    electron_transparency = 5,
    perlin = 6,
    diffuse_transparency = 7,
    checker = 8,
    goodsell = 9,
    surface_normal = 10
};

enum MaterialChameleonMode
{
    undefined_chameleon_mode = 0,
    emitter = 1,
    receiver = 2
};

enum OctreeDataType
{
    point,
    vector
};

#define OCTREE_DATA_OFFSET_X 0
#define OCTREE_DATA_OFFSET_Y 1
#define OCTREE_DATA_OFFSET_Z 2
#define OCTREE_DATA_SPACING_X 3
#define OCTREE_DATA_SPACING_Y 4
#define OCTREE_DATA_SPACING_Z 5
#define OCTREE_DATA_DIMENSION_X 6
#define OCTREE_DATA_DIMENSION_Y 7
#define OCTREE_DATA_DIMENSION_Z 8
#define OCTREE_DATA_INITIAL_DISTANCE 9
#define OCTREE_DATA_VALUES 10
#define OCTREE_DATA_INDICES 11

#define FIELD_VECTOR_DATA_SIZE 6
#define FIELD_VECTOR_OFFSET_POSITION_X 0
#define FIELD_VECTOR_OFFSET_POSITION_Y 1
#define FIELD_VECTOR_OFFSET_POSITION_Z 2
#define FIELD_VECTOR_OFFSET_DIRECTION_X 3
#define FIELD_VECTOR_OFFSET_DIRECTION_Y 4
#define FIELD_VECTOR_OFFSET_DIRECTION_Z 5

#define FIELD_POINT_DATA_SIZE 4
#define FIELD_POINT_OFFSET_POSITION_X 0
#define FIELD_POINT_OFFSET_POSITION_Y 1
#define FIELD_POINT_OFFSET_POSITION_Z 2
#define FIELD_POINT_OFFSET_VALUE 3

#define RAY_FLAG_PRIMARY 0
#define RAY_FLAG_SECONDARY 1
