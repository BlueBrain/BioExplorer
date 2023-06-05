/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#ifndef COMMONTYPES_H
#define COMMONTYPES_H

/** User data */
#define NO_USER_DATA -1

#define MATERIAL_PROPERTY_OPACITY "d"
#define MATERIAL_PROPERTY_MAP_OPACITY "map_d"
#define MATERIAL_PROPERTY_DIFFUSE_COLOR "kd"
#define MATERIAL_PROPERTY_MAP_DIFFUSE_COLOR "map_kd"
#define MATERIAL_PROPERTY_SPECULAR_COLOR "ks"
#define MATERIAL_PROPERTY_MAP_SPECULAR_COLOR "map_ks"
#define MATERIAL_PROPERTY_SPECULAR_INDEX "ns"
#define MATERIAL_PROPERTY_MAP_SPECULAR_INDEX "map_ns"
#define MATERIAL_PROPERTY_MAP_BUMP "map_bump"
#define MATERIAL_PROPERTY_REFRACTION "refraction"
#define MATERIAL_PROPERTY_MAP_REFRACTION "map_refraction"
#define MATERIAL_PROPERTY_REFLECTION "kr"
#define MATERIAL_PROPERTY_MAP_REFLECTION "map_kr"
#define MATERIAL_PROPERTY_EMISSION "a"
#define MATERIAL_PROPERTY_MAP_EMISSION "map_a"
#define MATERIAL_PROPERTY_SHADING_MODE "shading_mode"
#define MATERIAL_PROPERTY_USER_PARAMETER "user_parameter"
#define MATERIAL_PROPERTY_GLOSSINESS "glossiness"
#define MATERIAL_PROPERTY_CAST_USER_DATA "cast_user_data"
#define MATERIAL_PROPERTY_CLIPPING_MODE "clipping_mode"
#define MATERIAL_PROPERTY_CHAMELEON_MODE "chameleon_mode"
#define MATERIAL_PROPERTY_NODE_ID "node_id"
#define MATERIAL_PROPERTY_SKYBOX "skybox"
#define MATERIAL_PROPERTY_APPLY_SIMULATION "apply_simulation"

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
    goodsell = 9
};

enum MaterialChameleonMode
{
    undefined_chameleon_mode = 0,
    emitter = 1,
    receiver = 2
};

#endif
