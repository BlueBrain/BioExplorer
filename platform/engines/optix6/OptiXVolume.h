/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Daniel Nachbaur <daniel.nachbaur@epfl.ch>
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

#include <platform/core/common/transferFunction/TransferFunction.h>
#include <platform/core/engineapi/SharedDataVolume.h>

#include "OptiXModel.h"
#include "OptiXTypes.h"

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

namespace core
{
namespace engine
{
namespace optix
{
class OptiXVolume : public SharedDataVolume
{
public:
    OptiXVolume(OptiXModel* model, const Vector3ui& dimensions, const Vector3f& spacing, const DataType dataType,
                const VolumeParameters& params);

    void setDataRange(const Vector2f&) final{};
    void commit() final{};

    void setVoxels(const void* voxels) final;

protected:
    void _createBox(OptiXModel* model);

    float _getVoxelValue(const void* voxels, const uint16_t x, const uint16_t y, const uint16_t z) const;

    const VolumeParameters& _parameters;
    const Vector3f _offset{0.f, 0.f, 0.f};

    RTformat _dataType{RT_FORMAT_UNSIGNED_BYTE};
    uint64_t _dataTypeSize{1};

private:
    OptiXModel* _model{nullptr};
};
} // namespace optix
} // namespace engine
} // namespace core