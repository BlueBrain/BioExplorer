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
    odt_undefined = 0,
    odt_points = 1,
    odt_vectors = 2
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
