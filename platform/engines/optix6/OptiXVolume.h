/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
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

#include <platform/core/common/CommonTypes.h>
#include <platform/core/common/transferFunction/TransferFunction.h>
#include <platform/core/engineapi/SharedDataVolume.h>
#include <platform/core/engineapi/Volume.h>

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
class OptiXVolume : public virtual Volume
{
public:
    /** @copydoc Volume::Volume */
    OptiXVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType dataType,
                const VolumeParameters& params);

    /** @copydoc Volume::setDataRange */
    void setDataRange(const Vector2f&) final
    { /*Not applicable*/
    }

    /** @copydoc Volume::commit */
    void commit() final
    { /*Not applicable*/
    }

protected:
    const VolumeParameters& _parameters;

    RTformat _dataType{RT_FORMAT_UNSIGNED_BYTE};
    uint64_t _dataTypeSize{1};
};

class OptiXSharedDataVolume : public SharedDataVolume, public OptiXVolume
{
public:
    /** @copydoc SharedDataVolume::SharedDataVolume */
    OptiXSharedDataVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType dataType,
                          const VolumeParameters& params);

    /** @copydoc SharedDataVolume::setVoxels */
    void setVoxels(const void* voxels) final;
};
} // namespace optix
} // namespace engine
} // namespace core