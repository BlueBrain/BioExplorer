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

namespace brayns
{
class OptiXCamera;
using OptiXCameraPtr = std::shared_ptr<OptiXCamera>;

constexpr size_t OPTIX_STACK_SIZE = 4096;
constexpr size_t OPTIX_RAY_TYPE_COUNT = 2;
constexpr size_t OPTIX_ENTRY_POINT_COUNT = 1;

constexpr float EPSILON = 1e-2f;

const std::string CUDA_OUTPUT_BUFFER = "output_buffer";
const std::string CUDA_ACCUMULATION_BUFFER = "accum_buffer";
const std::string CUDA_DENOISED_BUFFER = "denoised_buffer";
const std::string CUDA_TONEMAPPED_BUFFER = "tonemapped_buffer";
const std::string CUDA_FRAME_NUMBER = "frame_number";

} // namespace brayns
