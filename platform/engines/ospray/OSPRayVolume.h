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
#include <platform/core/engineapi/BrickedVolume.h>
#include <platform/core/engineapi/SharedDataVolume.h>

#include <ospray/SDK/volume/Volume.h>

namespace core
{
namespace engine
{
namespace ospray
{
class OSPRayVolume : public virtual Volume
{
public:
    OSPRayVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType type,
                 const VolumeParameters& params, OSPTransferFunction transferFunction, const std::string& volumeType);
    ~OSPRayVolume();

    /** @copydoc Volume::setDataRange */
    void setDataRange(const Vector2f& range) final;

    /** @copydoc Volume::commit */
    void commit() final;

    OSPVolume impl() const { return _volume; }

protected:
    size_t _dataSize{0};
    const VolumeParameters& _parameters;
    OSPVolume _volume;
    OSPDataType _ospType;
};

class OSPRayBrickedVolume : public BrickedVolume, public OSPRayVolume
{
public:
    /** @copydoc BrickedVolume::BrickedVolume */
    OSPRayBrickedVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType type,
                        const VolumeParameters& params, OSPTransferFunction transferFunction);
    /** @copydoc SharedDataVolume::setBrick */
    void setBrick(const void* data, const Vector3ui& position, const Vector3ui& size) final;
};

class OSPRaySharedDataVolume : public SharedDataVolume, public OSPRayVolume
{
public:
    /** @copydoc SharedDataVolume::SharedDataVolume */
    OSPRaySharedDataVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType type,
                           const VolumeParameters& params, OSPTransferFunction transferFunction);

    /** @copydoc SharedDataVolume::setVoxels */
    void setVoxels(const void* voxels) final;
};
} // namespace ospray
} // namespace engine
} // namespace core