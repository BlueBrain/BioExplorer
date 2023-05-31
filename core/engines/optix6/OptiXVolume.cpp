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

#include "OptiXVolume.h"
#include "OptiXContext.h"
#include "OptiXMaterial.h"
#include "OptiXUtils.h"

#include <core/brayns/parameters/VolumeParameters.h>

namespace brayns
{
OptiXVolume::OptiXVolume(OptiXModel* model, const Vector3ui& dimensions, const Vector3f& spacing, const DataType type,
                         const VolumeParameters& params)
    : Volume(dimensions, spacing, type)
    , SharedDataVolume(dimensions, spacing, type)
    , _parameters(params)
{
    CORE_INFO("Volume dimension: " << _dimensions);
    CORE_INFO("Volume spacing  : " << _spacing);
    switch (type)
    {
    case DataType::INT8:
        _dataType = RT_FORMAT_BYTE;
        _dataTypeSize = 1;
        CORE_INFO("Volume data type: RT_FORMAT_BYTE");
        break;
    case DataType::UINT8:
        _dataType = RT_FORMAT_UNSIGNED_BYTE;
        _dataTypeSize = 1;
        CORE_INFO("Volume data type: RT_FORMAT_UNSIGNED_BYTE");
        break;
    case DataType::INT16:
        _dataType = RT_FORMAT_INT;
        _dataTypeSize = 2;
        CORE_INFO("Volume data type: RT_FORMAT_INT");
        break;
    case DataType::UINT16:
        _dataType = RT_FORMAT_UNSIGNED_INT;
        _dataTypeSize = 2;
        CORE_INFO("Volume data type: RT_FORMAT_UNSIGNED_INT");
        break;
    default:
        throw std::runtime_error("Unsupported voxel type " + std::to_string(int(type)));
    }

    auto context = OptiXContext::get().getOptixContext();
    context[CONTEXT_VOLUME_DATA_TYPE_SIZE]->setUint(_dataTypeSize);

    _createBox(model);
}

OptiXVolume::~OptiXVolume()
{
    auto context = OptiXContext::get().getOptixContext();
    _buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, 0);

    RT_DESTROY(_buffer);
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

    auto context = OptiXContext::get().getOptixContext();
    const auto bufferSize = _dimensions.x * _dimensions.y * _dimensions.z * _dataTypeSize;
    _buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, bufferSize);
    memcpy(_buffer->map(), voxels, bufferSize);
    _buffer->unmap();
    context[CONTEXT_VOLUME_DATA]->setBuffer(_buffer);
    context[CONTEXT_VOLUME_DIMENSIONS]->setUint(_dimensions.x, _dimensions.y, _dimensions.z);
    context[CONTEXT_VOLUME_OFFSET]->setFloat(_offset.x, _offset.y, _offset.z);
    context[CONTEXT_VOLUME_ELEMENT_SPACING]->setFloat(_spacing.x, _spacing.y, _spacing.z);
}

} // namespace brayns
