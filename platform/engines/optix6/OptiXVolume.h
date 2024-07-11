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