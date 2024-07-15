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

#include "OptiXVolume.h"
#include "OptiXContext.h"
#include "OptiXMaterial.h"
#include "OptiXUtils.h"

#include <platform/core/parameters/VolumeParameters.h>

using namespace optix;

namespace core
{
namespace engine
{
namespace optix
{

OptiXVolume::OptiXVolume(const Vector3ui& dimensions, const Vector3f& spacing, const DataType dataType,
                         const VolumeParameters& params)
    : Volume(dimensions, spacing, dataType)
    , _parameters(params)
{
    CORE_INFO("Volume dimension: " << _dimensions);
    CORE_INFO("Volume spacing  : " << _spacing);
    switch (dataType)
    {
    case DataType::INT8:
        _dataType = RT_FORMAT_BYTE;
        _dataTypeSize = sizeof(int8_t);
        CORE_INFO("Volume data type: RT_FORMAT_BYTE");
        break;
    case DataType::UINT8:
        _dataType = RT_FORMAT_UNSIGNED_BYTE;
        _dataTypeSize = sizeof(uint8_t);
        CORE_INFO("Volume data type: RT_FORMAT_UNSIGNED_BYTE");
        break;
    case DataType::INT16:
        _dataType = RT_FORMAT_INT;
        _dataTypeSize = sizeof(int16_t);
        CORE_INFO("Volume data type: RT_FORMAT_INT");
        break;
    case DataType::UINT16:
        _dataType = RT_FORMAT_UNSIGNED_INT;
        _dataTypeSize = sizeof(uint16_t);
        CORE_INFO("Volume data type: RT_FORMAT_UNSIGNED_INT");
        break;
    case DataType::FLOAT:
        _dataType = RT_FORMAT_FLOAT;
        _dataTypeSize = sizeof(float);
        CORE_INFO("Volume data type: RT_FORMAT_FLOAT");
        break;
    default:
        throw std::runtime_error("Unsupported voxel type " + std::to_string(int(dataType)));
    }
}

OptiXSharedDataVolume::OptiXSharedDataVolume(const Vector3ui& dimensions, const Vector3f& spacing,
                                             const DataType dataType, const VolumeParameters& params)
    : Volume(dimensions, spacing, dataType)
    , SharedDataVolume(dimensions, spacing, dataType)
    , OptiXVolume(dimensions, spacing, dataType, params)
{
}

void OptiXSharedDataVolume::setVoxels(const void* voxels)
{
    const uint64_t volumeSize = _dimensions.x * _dimensions.y * _dimensions.z;
    floats volumeAsFloats(volumeSize);
    _valueRange = Vector2f(1e6, -1e6);

    for (uint64_t i = 0; i < volumeSize; ++i)
    {
        switch (_dataType)
        {
        case RT_FORMAT_BYTE:
        {
            int8_t value;
            int8_t* v = (int8_t*)voxels;
            memcpy(&value, v + i, sizeof(int8_t));
            volumeAsFloats[i] = value;
            _valueRange.x = std::min(_valueRange.x, (float)value);
            _valueRange.y = std::max(_valueRange.y, (float)value);
            break;
        }
        case RT_FORMAT_UNSIGNED_BYTE:
        {
            uint8_t value;
            uint8_t* v = (uint8_t*)voxels;
            memcpy(&value, v + i, sizeof(uint8_t));
            volumeAsFloats[i] = value;
            _valueRange.x = std::min(_valueRange.x, (float)value);
            _valueRange.y = std::max(_valueRange.y, (float)value);
            break;
        }
        case RT_FORMAT_INT:
        {
            int16_t value;
            int16_t* v = (int16_t*)voxels;
            memcpy(&value, v + i, sizeof(int16_t));
            volumeAsFloats[i] = value;
            _valueRange.x = std::min(_valueRange.x, (float)value);
            _valueRange.y = std::max(_valueRange.y, (float)value);
            break;
        }
        case RT_FORMAT_UNSIGNED_INT:
        {
            uint16_t value;
            uint16_t* v = (uint16_t*)voxels;
            memcpy(&value, v + i, sizeof(uint16_t));
            volumeAsFloats[i] = value;
            _valueRange.x = std::min(_valueRange.x, (float)value);
            _valueRange.y = std::max(_valueRange.y, (float)value);
            break;
        }
        case RT_FORMAT_FLOAT:
        {
            float value;
            float* v = (float*)voxels;
            memcpy(&value, v + i, sizeof(float));
            volumeAsFloats[i] = value;
            _valueRange.x = std::min(_valueRange.x, (float)value);
            _valueRange.y = std::max(_valueRange.y, (float)value);
            break;
        }
        }
    }
    const size_t bufferSize = volumeAsFloats.size() * sizeof(float);
    _memoryBuffer.resize(bufferSize);
    memcpy(_memoryBuffer.data(), volumeAsFloats.data(), bufferSize);
}
} // namespace optix
} // namespace engine
} // namespace core