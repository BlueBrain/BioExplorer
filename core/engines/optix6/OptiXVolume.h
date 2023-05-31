/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Daniel Nachbaur <daniel.nachbaur@epfl.ch>
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

#include <core/brayns/common/transferFunction/TransferFunction.h>
#include <core/brayns/engineapi/SharedDataVolume.h>

#include "OptiXModel.h"
#include "OptiXTypes.h"

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

namespace brayns
{
class OptiXVolume : public SharedDataVolume
{
public:
    OptiXVolume(OptiXModel* model, const Vector3ui& dimensions, const Vector3f& spacing, const DataType type,
                const VolumeParameters& params);
    ~OptiXVolume();

    void setDataRange(const Vector2f&) final{};
    void commit() final{};

    void setVoxels(const void* voxels) final;

protected:
    void _createBox(OptiXModel* model);

    const VolumeParameters& _parameters;
    const Vector3f _offset{0.f, 0.f, 0.f};

    RTformat _dataType{RT_FORMAT_UNSIGNED_BYTE};
    uint64_t _dataTypeSize{1};
    ::optix::Buffer _buffer{nullptr};
};

} // namespace brayns
