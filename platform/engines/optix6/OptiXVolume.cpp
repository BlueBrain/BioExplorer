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
    // _createBox(model);
}

OptiXVolume::~OptiXVolume()
{
    RT_DESTROY(_buffer);
    RT_DESTROY(_sampler);
}

void OptiXVolume::_createBox(OptiXModel* model)
{
    const Vector3f positions[8] = {
        {0.f, 0.f, 0.f}, //
        {1.f, 0.f, 0.f}, //    6--------7
        {0.f, 1.f, 0.f}, //   /|       /|
        {1.f, 1.f, 0.f}, //  2--------3 |
        {0.f, 0.f, 1.f}, //  | |      | |
        {1.f, 0.f, 1.f}, //  | 4------|-5
        {0.f, 1.f, 1.f}, //  |/       |/
        {1.f, 1.f, 1.f}  //  0--------1
    };

    const uint16_t indices[6][6] = {
        {5, 4, 6, 6, 7, 5}, // Front
        {7, 5, 1, 1, 3, 7}, // Right
        {3, 1, 0, 0, 2, 3}, // Back
        {2, 0, 4, 4, 6, 2}, // Left
        {0, 1, 5, 5, 4, 0}, // Bottom
        {7, 3, 2, 2, 6, 7}  // Top
    };

    size_t materialId = 0;
    const Vector3f BLACK = {0.f, 0.f, 0.f};
    Boxd bounds;
    for (size_t i = 0; i < 6; ++i)
    {
        // Cornell box
        auto material = model->createMaterial(materialId, "box" + std::to_string(materialId));
        material->setDiffuseColor(BLACK);
        material->setSpecularColor(BLACK);
        material->setOpacity(0.f);

        auto& triangleMesh = model->getTriangleMeshes()[materialId];
        for (size_t j = 0; j < 6; ++j)
        {
            const auto position = Vector3f(_dimensions) * _spacing * positions[indices[i][j]];
            triangleMesh.vertices.push_back(position);
            bounds.merge(position);
        }
        triangleMesh.indices.push_back(Vector3ui(0, 1, 2));
        triangleMesh.indices.push_back(Vector3ui(3, 4, 5));
        ++materialId;
    }
    model->mergeBounds(bounds);
}

void OptiXVolume::setVoxels(const void* voxels)
{
    RT_DESTROY(_buffer);
    RT_DESTROY(_sampler);

    auto context = OptiXContext::get().getOptixContext();
    context[CONTEXT_VOLUME_DIMENSIONS]->setUint(_dimensions.x, _dimensions.y, _dimensions.z);
    context[CONTEXT_VOLUME_OFFSET]->setFloat(_offset.x, _offset.y, _offset.z);
    context[CONTEXT_VOLUME_ELEMENT_SPACING]->setFloat(_spacing.x, _spacing.y, _spacing.z);

    optix::Buffer _buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _dimensions.x,
                                                           _dimensions.y, _dimensions.z, 1u);
    const uint64_t volumeSize = _dimensions.x * _dimensions.y * _dimensions.z;
    float* volumeAsFloats = (float*)_buffer->map();
    _dataRange = Vector2f(1e6, -1e6);

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
            _dataRange.x = std::min(_dataRange.x, (float)value);
            _dataRange.y = std::max(_dataRange.y, (float)value);
            break;
        }
        case RT_FORMAT_UNSIGNED_BYTE:
        {
            uint8_t value;
            uint8_t* v = (uint8_t*)voxels;
            memcpy(&value, v + i, sizeof(uint8_t));
            volumeAsFloats[i] = value;
            _dataRange.x = std::min(_dataRange.x, (float)value);
            _dataRange.y = std::max(_dataRange.y, (float)value);
            break;
        }
        case RT_FORMAT_INT:
        {
            int16_t value;
            int16_t* v = (int16_t*)voxels;
            memcpy(&value, v + i, sizeof(int16_t));
            volumeAsFloats[i] = value;
            _dataRange.x = std::min(_dataRange.x, (float)value);
            _dataRange.y = std::max(_dataRange.y, (float)value);
            break;
        }
        case RT_FORMAT_UNSIGNED_INT:
        {
            uint16_t value;
            uint16_t* v = (uint16_t*)voxels;
            memcpy(&value, v + i, sizeof(uint16_t));
            volumeAsFloats[i] = value;
            _dataRange.x = std::min(_dataRange.x, (float)value);
            _dataRange.y = std::max(_dataRange.y, (float)value);
            break;
        }
        case RT_FORMAT_FLOAT:
        {
            float value;
            float* v = (float*)voxels;
            memcpy(&value, v + i, sizeof(float));
            volumeAsFloats[i] = value;
            _dataRange.x = std::min(_dataRange.x, (float)value);
            _dataRange.y = std::max(_dataRange.y, (float)value);
            break;
        }
        }
    }
    _buffer->unmap();

    // Volume as texture
    const size_t materialId = 0;
    auto material = static_cast<OptiXMaterial*>(_model->getMaterial(materialId).get());
    auto& textureSamplers = material->getTextureSamplers();
    const auto it = textureSamplers.find(TextureType::volume);
    if (it != textureSamplers.end())
        textureSamplers.erase(it);
    _sampler = context->createTextureSampler();
    _sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    _sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    _sampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    _sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    _sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    _sampler->setBuffer(0u, 0u, _buffer);
    _sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    textureSamplers.insert(std::make_pair(TextureType::volume, _sampler));
    auto optixMaterial = material->getOptixMaterial();
    const auto textureName = textureTypeToString[static_cast<uint8_t>(TextureType::volume)];
    CORE_INFO("Registering " + textureName + " texture");
    optixMaterial[textureName]->setInt(_sampler->getId());
    material->commit();

    const auto textureSamplerId = _sampler->getId();
    auto& volumeGeometries = _model->getVolumeGeometries();
    volumeGeometries[materialId].textureSamplerId = textureSamplerId;
    _model->commitVolumesBuffers(materialId);

    context[CONTEXT_VOLUME_TEXTURE_SAMPLER]->setInt(textureSamplerId);
    context[CONTEXT_VOLUME_DATA_TYPE]->setUint(_dataType);

    CORE_INFO("Volume range: " << _dataRange);
}

} // namespace core
