/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include <memory>

namespace brayns
{
class OptiXCamera;
using OptiXCameraPtr = std::shared_ptr<OptiXCamera>;
class OptiXCameraProgram;
using OptiXCameraProgramPtr = std::shared_ptr<OptiXCameraProgram>;

constexpr size_t OPTIX_STACK_SIZE = 16384;
constexpr size_t OPTIX_RAY_TYPE_COUNT = 2;
constexpr size_t OPTIX_ENTRY_POINT_COUNT = 1;

constexpr float EPSILON = 1e-2f;

} // namespace brayns
