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

#include "OptiXVolume.h"
#include "OptiXContext.h"
#include "OptiXMaterial.h"
#include "OptiXUtils.h"

#include <platform/core/parameters/VolumeParameters.h>

using namespace optix;

namespace core
{
OptiXVolume::OptiXVolume(OptiXModel* model, const Vector3ui& dimensions, const Vector3f& spacing,
                         const DataType dataType, const VolumeParameters& params)
    : Volume(dimensions, spacing, dataType)
    , SharedDataVolume(dimensions, spacing, dataType)
    , _model(model)
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

void OptiXVolume::setVoxels(const void* voxels)
{
    auto context = OptiXContext::get().getOptixContext();
    Buffer buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _dimensions.x, _dimensions.y,
                                                   _dimensions.z, 1u);
    const uint64_t volumeSize = _dimensions.x * _dimensions.y * _dimensions.z;
    float* volumeAsFloats = (float*)buffer->map();
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
    buffer->unmap();

    // Volume as texture
    const size_t materialId = VOLUME_MATERIAL_ID;
    auto material = static_cast<OptiXMaterial*>(_model->getMaterial(materialId).get());
    material->setValueRange(_valueRange);
    auto& textureSamplers = material->getTextureSamplers();
    const auto it = textureSamplers.find(TextureType::volume);
    if (it != textureSamplers.end())
        textureSamplers.erase(it);

    TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    sampler->setBuffer(0u, 0u, buffer);
    sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    textureSamplers.insert(std::make_pair(TextureType::volume, sampler));
    auto optixMaterial = material->getOptixMaterial();
    const auto textureName = textureTypeToString[static_cast<uint8_t>(TextureType::volume)];
    optixMaterial[textureName]->setInt(sampler->getId());
    material->commit();

    const auto volumeSamplerId = sampler->getId();
    auto& volumeGeometries = _model->getVolumeGeometries();

    volumeGeometries[materialId].volumeSamplerId = volumeSamplerId;
    volumeGeometries[materialId].valueRange = _valueRange;
    _model->commitVolumesBuffers(materialId);
}

} // namespace core
