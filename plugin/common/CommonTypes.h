/* Copyright (c) 2020-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: cyrille.favreau@epfl.ch
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

#define DEFAULT_SKY_POWER 4.f

/** Additional material attributes */
#define MATERIAL_PROPERTY_SHADING_MODE "shading_mode"
#define MATERIAL_PROPERTY_USER_PARAMETER "user_parameter"

enum MaterialShadingMode
{
    none = 0,
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

enum CameraStereoMode
{
    mono = 0,
    left = 1,
    right = 2,
    side_by_side = 3
};
