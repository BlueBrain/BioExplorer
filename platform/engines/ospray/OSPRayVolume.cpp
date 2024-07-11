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

#include "OSPRayVolume.h"
#include "OSPRayProperties.h"
#include "OSPRayUtils.h"

#include <platform/core/common/Properties.h>
#include <platform/core/parameters/VolumeParameters.h>

namespace core
{
namespace engine
{
namespace ospray
{
OSPRayVolume::OSPRayVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType type,
                           const VolumeParameters& params, OSPTransferFunction transferFunction,
                           const std::string& volumeType)
    : Volume(dimensions, spacing, type)
    , _parameters(params)
    , _volume(ospNewVolume(volumeType.c_str()))
{
    osphelper::set(_volume, OSPRAY_VOLUME_PROPERTY_DIMENSIONS, Vector3i(dimensions));
    osphelper::set(_volume, OSPRAY_VOLUME_PROPERTY_GRID_SPACING, Vector3f(spacing));

    switch (type)
    {
    case DataType::FLOAT:
        osphelper::set(_volume, OSPRAY_VOLUME_VOXEL_TYPE, "float");
        _ospType = OSP_FLOAT;
        _dataSize = 4;
        break;
    case DataType::DOUBLE:
        osphelper::set(_volume, OSPRAY_VOLUME_VOXEL_TYPE, "double");
        _ospType = OSP_DOUBLE;
        _dataSize = 8;
        break;
    case DataType::UINT8:
        osphelper::set(_volume, OSPRAY_VOLUME_VOXEL_TYPE, "uchar");
        _ospType = OSP_UINT;
        _dataSize = 1;
        break;
    case DataType::UINT16:
        osphelper::set(_volume, OSPRAY_VOLUME_VOXEL_TYPE, "ushort");
        _ospType = OSP_UINT2;
        _dataSize = 2;
        break;
    case DataType::INT16:
        osphelper::set(_volume, OSPRAY_VOLUME_VOXEL_TYPE, "short");
        _ospType = OSP_INT2;
        _dataSize = 2;
        break;
    case DataType::UINT32:
    case DataType::INT8:
    case DataType::INT32:
        throw std::runtime_error("Unsupported voxel type " + std::to_string(int(type)));
    }

    ospSetObject(_volume, DEFAULT_COMMON_TRANSFER_FUNCTION, transferFunction);
}

OSPRayVolume::~OSPRayVolume()
{
    ospRelease(_volume);
}

OSPRayBrickedVolume::OSPRayBrickedVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType type,
                                         const VolumeParameters& params, OSPTransferFunction transferFunction)
    : Volume(dimensions, spacing, type)
    , BrickedVolume(dimensions, spacing, type)
    , OSPRayVolume(dimensions, spacing, type, params, transferFunction, OSPRAY_VOLUME_PROPERTY_TYPE_BLOCK_BRICKED)
{
}

OSPRaySharedDataVolume::OSPRaySharedDataVolume(const Vector3ui& dimensions, const Vector3f& spacing,
                                               const DataType type, const VolumeParameters& params,
                                               OSPTransferFunction transferFunction)
    : Volume(dimensions, spacing, type)
    , SharedDataVolume(dimensions, spacing, type)
    , OSPRayVolume(dimensions, spacing, type, params, transferFunction, OSPRAY_VOLUME_PROPERTY_TYPE_SHARED_STRUCTURED)
{
}

void OSPRayVolume::setDataRange(const Vector2f& range)
{
    _valueRange = range;
    osphelper::set(_volume, OSPRAY_VOLUME_VOXEL_RANGE, _valueRange);
    markModified();
}

void OSPRayBrickedVolume::setBrick(const void* data, const Vector3ui& position, const Vector3ui& size_)
{
    const ospcommon::vec3i pos{int(position.x), int(position.y), int(position.z)};
    const ospcommon::vec3i size{int(size_.x), int(size_.y), int(size_.z)};
    ospSetRegion(_volume, const_cast<void*>(data), (osp::vec3i&)pos, (osp::vec3i&)size);
    BrickedVolume::_sizeInBytes += glm::compMul(size_) * _dataSize;
    markModified();
}

void OSPRaySharedDataVolume::setVoxels(const void* voxels)
{
    OSPData data = ospNewData(glm::compMul(SharedDataVolume::_dimensions), _ospType, voxels, OSP_DATA_SHARED_BUFFER);
    SharedDataVolume::_sizeInBytes += glm::compMul(SharedDataVolume::_dimensions) * _dataSize;
    ospSetData(_volume, OSPRAY_VOLUME_VOXEL_DATA, data);
    ospRelease(data);
    markModified();
}

void OSPRayVolume::commit()
{
    if (_parameters.isModified())
    {
        osphelper::set(_volume, OSPRAY_VOLUME_GRADIENT_SHADING_ENABLED, _parameters.getGradientShading());
        osphelper::set(_volume, OSPRAY_VOLUME_GRADIENT_OFFSET, static_cast<float>(_parameters.getGradientOffset()));
        osphelper::set(_volume, OSPRAY_VOLUME_ADAPTIVE_MAX_SAMPLING_RATE,
                       static_cast<float>(_parameters.getAdaptiveMaxSamplingRate()));
        osphelper::set(_volume, OSPRAY_VOLUME_ADAPTIVE_SAMPLING, _parameters.getAdaptiveSampling());
        osphelper::set(_volume, OSPRAY_VOLUME_SINGLE_SHADE, _parameters.getSingleShade());
        osphelper::set(_volume, OSPRAY_VOLUME_PRE_INTEGRATION, _parameters.getPreIntegration());
        osphelper::set(_volume, OSPRAY_VOLUME_SAMPLING_RATE, static_cast<float>(_parameters.getSamplingRate()));
        osphelper::set(_volume, OSPRAY_VOLUME_SPECULAR_EXPONENT, Vector3f(_parameters.getSpecular()));
        osphelper::set(_volume, OSPRAY_VOLUME_USER_PARAMETERS, Vector3f(_parameters.getUserParameters()));
        osphelper::set(_volume, OSPRAY_VOLUME_CLIPPING_BOX_LOWER, Vector3f(_parameters.getClipBox().getMin()));
        osphelper::set(_volume, OSPRAY_VOLUME_CLIPPING_BOX_UPPER, Vector3f(_parameters.getClipBox().getMax()));
    }
    if (isModified() || _parameters.isModified())
        ospCommit(_volume);
    resetModified();
}

} // namespace ospray
} // namespace engine
} // namespace core