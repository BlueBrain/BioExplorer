/*
 * Copyright (c) 2019-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
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

#include <platform/engines/optix6/cuda/Context.cuh>

static __device__ inline float4 calcTransferFunctionColor(const int sampleId, const float2& valueRange,
                                                          const float value)
{
    const float texcoord = (value - valueRange.x) / (valueRange.y - valueRange.x);
    return optix::rtTex1D<float4>(sampleId, texcoord);
}

/**
 * @brief Get the User Data object
 *
 * Note that the user data index to the user data buffer is currently stored in the x coordinates
 *
 * @return Color of the user data value after transfer function is applied
 */
static __device__ inline float4 getUserData()
{
    float4 color = make_float4(0.f);
    if (cast_user_data && userDataBuffer.size() > 0)
    {
        const ulong idx = userDataIndex;
        color = (idx >= userDataBuffer.size())
                    ? bad_color
                    : calcTransferFunctionColor(transfer_function_map, value_range, userDataBuffer[idx]);
    }
    return color;
}
