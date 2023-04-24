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

#include <brayns/parameters/VolumeParameters.h>
#include <engines/ospray/utils.h>

namespace brayns
{
OptiXVolume::OptiXVolume(OptiXModel* model, const Vector3ui& dimensions, const Vector3f& spacing, const DataType type,
                         const VolumeParameters& params)
    : Volume(dimensions, spacing, type)
    , SharedDataVolume(dimensions, spacing, type)
    , _parameters(params)
{
    BRAYNS_INFO("Volume dimension: " << _dimensions);
    BRAYNS_INFO("Volume spacing  : " << _spacing);
    switch (type)
    {
    case DataType::INT8:
        _dataType = RT_FORMAT_BYTE;
        _dataSize = 1;
        BRAYNS_INFO("Volume data type: RT_FORMAT_BYTE");
        break;
    case DataType::UINT8:
        _dataType = RT_FORMAT_UNSIGNED_BYTE;
        _dataSize = 1;
        BRAYNS_INFO("Volume data type: RT_FORMAT_UNSIGNED_BYTE");
        break;
    case DataType::INT16:
        _dataType = RT_FORMAT_INT;
        _dataSize = 2;
        BRAYNS_INFO("Volume data type: RT_FORMAT_INT");
        break;
    case DataType::UINT16:
        _dataType = RT_FORMAT_UNSIGNED_INT;
        _dataSize = 2;
        BRAYNS_INFO("Volume data type: RT_FORMAT_UNSIGNED_INT");
        break;
    default:
        throw std::runtime_error("Unsupported voxel type " + std::to_string(int(type)));
    }

    _bufferSize = dimensions.x * dimensions.y * dimensions.z * _dataSize;
    auto context = OptiXContext::get().getOptixContext();
    context["volumeDataSize"]->setUint(_dataSize);

    _createBox(model);
}

OptiXVolume::~OptiXVolume()
{
    if (_buffer)
        _buffer->destroy();
    if (_colorMapBuffer)
        _colorMapBuffer->destroy();
}

void OptiXVolume::_createBox(OptiXModel* model)
{
    const Vector3f positions[8] = {
        {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, //    6--------7
        {0.f, 1.f, 0.f},                  //   /|       /|
        {1.f, 1.f, 0.f},                  //  2--------3 |
        {0.f, 0.f, 1.f},                  //  | |      | |
        {1.f, 0.f, 1.f},                  //  | 4------|-5
        {0.f, 1.f, 1.f},                  //  |/       |/
        {1.f, 1.f, 1.f}                   //  0--------1
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
        material->setOpacity(0.0f);

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

void OptiXVolume::setDataRange(const Vector2f& range)
{
    auto context = OptiXContext::get().getOptixContext();
    context["colorMapMinValue"]->setFloat(range.x);
    context["colorMapRange"]->setFloat(range.y - range.x);
}

void OptiXVolume::setVoxels(const void* voxels)
{
    if (_buffer)
        _buffer->destroy();

    auto context = OptiXContext::get().getOptixContext();

    _buffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, _bufferSize);
    memcpy(_buffer->map(), voxels, _bufferSize);
    _buffer->unmap();
    context["volumeData"]->setBuffer(_buffer);
    context["volumeDimensions"]->setUint(_dimensions.x, _dimensions.y, _dimensions.z);
    context["volumeOffset"]->setFloat(_offset.x, _offset.y, _offset.z);
    context["volumeElementSpacing"]->setFloat(_spacing.x, _spacing.y, _spacing.z);
}

void OptiXVolume::commit()
{
    auto context = OptiXContext::get().getOptixContext();
    if (_parameters.isModified())
    {
        context["volumeGradientShadingEnabled"]->setUint(_parameters.getGradientShading());
        context["volumeAdaptiveMaxSamplingRate"]->setFloat(_parameters.getAdaptiveMaxSamplingRate());
        context["volumeAdaptiveSampling"]->setUint(_parameters.getAdaptiveSampling());
        context["volumeSingleShade"]->setUint(_parameters.getSingleShade());
        context["volumePreIntegration"]->setUint(_parameters.getPreIntegration());
        context["volumeSamplingRate"]->setFloat(_parameters.getSamplingRate());
        const Vector3f specular = _parameters.getSpecular();
        context["volumeSpecular"]->setFloat(specular.x, specular.y, specular.z);

        // context["volumeClippingBoxLower"]->setFloat(_parameters.getClipBox().getMin());
        // context["volumeClippingBoxUpper"]->setFloat(_parameters.getClipBox().getMax());
    }

    if (isModified() || _parameters.isModified())
    {
        // _commitTransferFunction();
    }
    resetModified();
}
} // namespace brayns
