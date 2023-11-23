/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include "OptiXModel.h"
#include "OptiXCommonStructs.h"
#include "OptiXContext.h"
#include "OptiXMaterial.h"
#include "OptiXUtils.h"
#include "OptiXVolume.h"

#include <platform/core/common/Logs.h>
#include <platform/core/common/simulation/AbstractSimulationHandler.h>
#include <platform/core/common/utils/Utils.h>
#include <platform/core/parameters/AnimationParameters.h>

#include <platform/core/engineapi/Material.h>

using namespace optix;

namespace core
{
namespace engine
{
namespace optix
{
template <typename T>
void setBufferRaw(RTbuffertype bufferType, RTformat bufferFormat, ::optix::Handle<::optix::BufferObj>& buffer,
                  ::optix::Handle<::optix::VariableObj> geometry, T* src, const size_t numElements, const size_t bytes)
{
    auto context = OptiXContext::get().getOptixContext();
    if (!buffer)
        buffer = context->createBuffer(bufferType, bufferFormat, numElements);
    else
        buffer->setSize(numElements);
    if (src != nullptr && numElements > 0 && bytes > 0)
    {
        memcpy(buffer->map(0, RT_BUFFER_MAP_WRITE_DISCARD), src, bytes);
        buffer->unmap();
    }
    geometry->setBuffer(buffer);
}

template <typename T>
void setBuffer(RTbuffertype bufferType, RTformat bufferFormat, ::optix::Handle<::optix::BufferObj>& buffer,
               ::optix::Handle<::optix::VariableObj> geometry, const std::vector<T>& src, const size_t numElements)
{
    setBufferRaw(bufferType, bufferFormat, buffer, geometry, src.data(), numElements, sizeof(T) * src.size());
}

OptiXModel::OptiXModel(AnimationParameters& animationParameters, VolumeParameters& volumeParameters,
                       GeometryParameters& geometryParameters)
    : Model(animationParameters, volumeParameters, geometryParameters)
{
}

OptiXModel::~OptiXModel()
{
    RT_DESTROY_MAP(_optixSpheres);
    RT_DESTROY_MAP(_optixCylinders)
    RT_DESTROY_MAP(_optixCones)
    RT_DESTROY_MAP(_optixVolumes)
    RT_DESTROY_MAP(_optixMeshes)
    RT_DESTROY_MAP(_optixStreamlines)
    RT_DESTROY_MAP(_optixTextures)
    RT_DESTROY_MAP(_optixTextureSamplers)
    for (auto optixMeshBuffer : _optixMeshBuffers)
    {
        RT_DESTROY(optixMeshBuffer.second.vertices_buffer);
        RT_DESTROY(optixMeshBuffer.second.normal_buffer);
        RT_DESTROY(optixMeshBuffer.second.texcoord_buffer);
        RT_DESTROY(optixMeshBuffer.second.indices_buffer);
    }
    RT_DESTROY(_geometryGroup);
    RT_DESTROY(_boundingBoxGroup);
}

void OptiXModel::commitGeometry()
{
    const auto compactBVH = getBVHFlags().count(BVHFlag::compact) > 0;
    auto& context = OptiXContext::get();
    // Geometry group
    if (!_geometryGroup)
        _geometryGroup = context.createGeometryGroup(compactBVH);

    // Bounding box group
    if (!_boundingBoxGroup)
        _boundingBoxGroup = context.createGeometryGroup(compactBVH);

    // Materials
    _commitMaterials();

    if (_spheresDirty)
        for (const auto& spheres : _geometries->_spheres)
            _commitSpheres(spheres.first);

    if (_cylindersDirty)
        for (const auto& cylinders : _geometries->_cylinders)
            _commitCylinders(cylinders.first);

    if (_conesDirty)
        for (const auto& cones : _geometries->_cones)
            _commitCones(cones.first);

    if (_triangleMeshesDirty)
        for (const auto& meshes : _geometries->_triangleMeshes)
            _commitMeshes(meshes.first);

    if (_volumesDirty)
        for (const auto& volume : _geometries->_volumes)
            _commitVolumes(volume.first);

    if (_streamlinesDirty)
        for (const auto& streamlines : _geometries->_streamlines)
            _commitStreamlines(streamlines.first);

    updateBounds();
    _markGeometriesClean();

    // handled by the scene
    _instancesDirty = false;

    CORE_DEBUG("Geometry group has " << _geometryGroup->getChildCount() << " children instances");
    CORE_DEBUG("Bounding box group has " << _boundingBoxGroup->getChildCount() << " children instances");
}

void OptiXModel::_commitSpheres(const size_t materialId)
{
    if (_geometries->_spheres.find(materialId) == _geometries->_spheres.end())
        return;

    auto context = OptiXContext::get().getOptixContext();
    const auto& spheres = _geometries->_spheres[materialId];
    context[CONTEXT_SPHERE_SIZE]->setUint(sizeof(Sphere) / sizeof(float));

    // Geometry
    _optixSpheres[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::sphere);
    _optixSpheres[materialId]->setPrimitiveCount(spheres.size());

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _spheresBuffers[materialId],
              _optixSpheres[materialId][OPTIX_GEOMETRY_PROPERTY_SPHERES], spheres, sizeof(Sphere) * spheres.size());

    // Material
    auto& mat = static_cast<OptiXMaterial&>(*_materials[materialId]);
    const auto material = mat.getOptixMaterial();
    if (!material)
        CORE_THROW(std::runtime_error("Material is not defined"));

    // Instance
    auto instance = context->createGeometryInstance();
    instance->setGeometry(_optixSpheres[materialId]);
    instance->setMaterialCount(1);
    instance->setMaterial(0, material);
    if (materialId == BOUNDINGBOX_MATERIAL_ID)
        _boundingBoxGroup->addChild(instance);
    else
        _geometryGroup->addChild(instance);
}

void OptiXModel::_commitCylinders(const size_t materialId)
{
    if (_geometries->_cylinders.find(materialId) == _geometries->_cylinders.end())
        return;

    auto context = OptiXContext::get().getOptixContext();
    const auto& cylinders = _geometries->_cylinders[materialId];
    context[CONTEXT_CYLINDER_SIZE]->setUint(sizeof(Cylinder) / sizeof(float));
    _optixCylinders[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::cylinder);

    auto& optixCylinders = _optixCylinders[materialId];
    optixCylinders->setPrimitiveCount(cylinders.size());

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _cylindersBuffers[materialId],
              _optixCylinders[materialId][OPTIX_GEOMETRY_PROPERTY_CYLINDERS], cylinders,
              sizeof(Cylinder) * cylinders.size());

    auto& mat = static_cast<OptiXMaterial&>(*_materials[materialId]);
    const auto material = mat.getOptixMaterial();
    if (!material)
        CORE_THROW(std::runtime_error("Material is not defined"));

    auto instance = context->createGeometryInstance();
    instance->setGeometry(optixCylinders);
    instance->setMaterialCount(1);
    instance->setMaterial(0, material);
    if (materialId == BOUNDINGBOX_MATERIAL_ID)
        _boundingBoxGroup->addChild(instance);
    else
        _geometryGroup->addChild(instance);
}

void OptiXModel::_commitCones(const size_t materialId)
{
    if (_geometries->_cones.find(materialId) == _geometries->_cones.end())
        return;

    auto context = OptiXContext::get().getOptixContext();
    const auto& cones = _geometries->_cones[materialId];
    context[CONTEXT_CONE_SIZE]->setUint(sizeof(Cone) / sizeof(float));
    _optixCones[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::cone);

    auto& optixCones = _optixCones[materialId];
    optixCones->setPrimitiveCount(cones.size());

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _conesBuffers[materialId],
              _optixCones[materialId][OPTIX_GEOMETRY_PROPERTY_CONES], cones, sizeof(Cone) * cones.size());

    auto& mat = static_cast<OptiXMaterial&>(*_materials[materialId]);
    auto material = mat.getOptixMaterial();
    if (!material)
        CORE_THROW(std::runtime_error("Material is not defined"));

    auto instance = context->createGeometryInstance();
    instance->setGeometry(optixCones);
    instance->setMaterialCount(1);
    instance->setMaterial(0, material);
    if (materialId == BOUNDINGBOX_MATERIAL_ID)
        _boundingBoxGroup->addChild(instance);
    else
        _geometryGroup->addChild(instance);
}

void OptiXModel::_commitMeshes(const size_t materialId)
{
    if (_geometries->_triangleMeshes.find(materialId) == _geometries->_triangleMeshes.end())
        return;

    const auto& meshes = _geometries->_triangleMeshes[materialId];
    _optixMeshes[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::triangleMesh);

    auto& optixMeshes = _optixMeshes[materialId];
    optixMeshes->setPrimitiveCount(meshes.indices.size());

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, _optixMeshBuffers[materialId].vertices_buffer,
              _optixMeshes[materialId][OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_VERTEX], meshes.vertices,
              meshes.vertices.size());

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, _optixMeshBuffers[materialId].indices_buffer,
              _optixMeshes[materialId][OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_INDEX], meshes.indices,
              meshes.indices.size());

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, _optixMeshBuffers[materialId].normal_buffer,
              _optixMeshes[materialId][OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_NORMAL], meshes.normals,
              meshes.normals.size());

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, _optixMeshBuffers[materialId].texcoord_buffer,
              _optixMeshes[materialId][OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_TEXTURE_COORDINATES],
              meshes.textureCoordinates, meshes.textureCoordinates.size());

    auto& mat = static_cast<OptiXMaterial&>(*_materials[materialId]);
    auto material = mat.getOptixMaterial();
    if (!material)
        CORE_THROW(std::runtime_error("Material is not defined"));

    auto context = OptiXContext::get().getOptixContext();
    auto instance = context->createGeometryInstance();
    instance->setGeometry(optixMeshes);
    instance->setMaterialCount(1);
    instance->setMaterial(0, material);
    if (materialId == BOUNDINGBOX_MATERIAL_ID)
        _boundingBoxGroup->addChild(instance);
    else
        _geometryGroup->addChild(instance);
}

void OptiXModel::_commitStreamlines(const size_t materialId)
{
    if (_geometries->_streamlines.find(materialId) == _geometries->_streamlines.end())
        return;

    const auto& streamlines = _geometries->_streamlines[materialId];

    // Identify streamlines according to indices
    std::vector<Vector2ui> indices;
    size_t begin = 0;
    for (size_t i = 0; i < streamlines.indices.size() - 1; ++i)
    {
        if (streamlines.indices[i] + 1 != streamlines.indices[i + 1])
        {
            indices.push_back({begin, streamlines.indices[i]});
            begin = streamlines.indices[i + 1];
        }
    }

    Vector4fs vertexCurve;
    Vector4fs colorCurve = streamlines.vertexColor;
    uint32_ts indexCurve;
    const bool processColor = colorCurve.empty();
    size_t count = 0;
    for (const auto& index : indices)
    {
        const uint32_t begin = index.x;
        const uint32_t end = index.y;

        Vector4fs controlPoints;
        for (uint32_t idx = begin; idx < end; ++idx)
            controlPoints.push_back(streamlines.vertex[idx]);

        const auto lengthSegment = length(Vector3f(streamlines.vertex[end]) - Vector3f(streamlines.vertex[begin]));
        const auto nbSteps = max(2, floor(lengthSegment / 3.f));
        const float t_step = 1.f / nbSteps;
        for (float t = 0.f; t < 1.f - t_step; t += t_step)
        {
            const auto a = getBezierPoint(controlPoints, t);
            const auto b = getBezierPoint(controlPoints, t + t_step);
            const auto src = Vector3f(a);
            const auto dst = Vector3f(b);
            const auto srcRadius = a.w;
            const auto dstRadius = b.w;
            if (t == 0.f)
            {
                vertexCurve.push_back(Vector4f(src, 0.f));
                vertexCurve.push_back(Vector4f(dst, dstRadius));
            }
            else if (t == 1.f - t_step)
            {
                vertexCurve.push_back(Vector4f(src, srcRadius));
                vertexCurve.push_back(Vector4f(dst, 0.f));
            }
            else
            {
                vertexCurve.push_back(Vector4f(src, srcRadius));
                vertexCurve.push_back(Vector4f(dst, dstRadius));
            }

            if (processColor)
                colorCurve.push_back(Vector4f(0.5f + 0.5f * normalize(dst - src), 1.f));

            indexCurve.push_back(count);
            count += 2;
        }
    }

    _optixStreamlines[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::streamline);

    auto& optixStreamlines = _optixStreamlines[materialId];
    optixStreamlines->setPrimitiveCount(indexCurve.size());

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, _streamlinesBuffers[materialId].vertices_buffer,
              _optixStreamlines[materialId][OPTIX_GEOMETRY_PROPERTY_STREAMLINE_VERTEX], vertexCurve,
              vertexCurve.size());

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, _streamlinesBuffers[materialId].indices_buffer,
              _optixStreamlines[materialId][OPTIX_GEOMETRY_PROPERTY_STREAMLINE_MESH_INDEX], indexCurve,
              indexCurve.size());

    auto& material = static_cast<OptiXMaterial&>(*_materials[materialId]);
    auto optixMaterial = material.getOptixMaterial();
    if (!optixMaterial)
        CORE_THROW(std::runtime_error("OptiX material is not defined"));

    auto context = OptiXContext::get().getOptixContext();

    const Vector2ui textureSize{MAX_TEXTURE_SIZE, 1 + colorCurve.size() / MAX_TEXTURE_SIZE};
    Buffer buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, textureSize.x, textureSize.y, 1u);
    memcpy(buffer->map(), colorCurve.data(), sizeof(Vector4f) * colorCurve.size());
    buffer->unmap();

    TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    sampler->setBuffer(0u, 0u, buffer);
    sampler->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
    sampler->setMaxAnisotropy(8.0f);
    sampler->validate();
    const auto samplerId = sampler->getId();
    auto& textureSamplers = material.getTextureSamplers();
    textureSamplers.insert(std::make_pair(TextureType::diffuse, sampler));
    const auto textureName = textureTypeToString[static_cast<uint8_t>(TextureType::diffuse)];
    optixMaterial[textureName]->setInt(samplerId);
    material.commit();

    auto instance = context->createGeometryInstance();
    instance->setGeometry(optixStreamlines);
    instance->setMaterialCount(1);
    instance->setMaterial(0, optixMaterial);
    if (materialId == BOUNDINGBOX_MATERIAL_ID)
        _boundingBoxGroup->addChild(instance);
    else
        _geometryGroup->addChild(instance);
}

void OptiXModel::_commitMaterials()
{
    CORE_DEBUG("Committing " << _materials.size() << " OptiX materials");

    for (auto& material : _materials)
        material.second->commit();
}

void OptiXModel::buildBoundingBox()
{
    if (_boundingBoxBuilt)
        return;

    _boundingBoxBuilt = true;

    auto material = createMaterial(BOUNDINGBOX_MATERIAL_ID, "bounding_box");
    material->setDiffuseColor({1, 1, 1});
    material->setEmission(1.f);

    const Vector3f s(0.5f);
    const Vector3f c(0.5f);
    const float radius = 0.005f;
    const Vector3f positions[8] = {
        {c.x - s.x, c.y - s.y, c.z - s.z}, {c.x + s.x, c.y - s.y, c.z - s.z}, //    6--------7
        {c.x - s.x, c.y + s.y, c.z - s.z},                                    //   /|       /|
        {c.x + s.x, c.y + s.y, c.z - s.z},                                    //  2--------3 |
        {c.x - s.x, c.y - s.y, c.z + s.z},                                    //  | |      | |
        {c.x + s.x, c.y - s.y, c.z + s.z},                                    //  | 4------|-5
        {c.x - s.x, c.y + s.y, c.z + s.z},                                    //  |/       |/
        {c.x + s.x, c.y + s.y, c.z + s.z}                                     //  0--------1
    };

    for (size_t i = 0; i < 8; ++i)
        addSphere(BOUNDINGBOX_MATERIAL_ID, Sphere(positions[i], radius));

    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[0], positions[1], radius});
    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[2], positions[3], radius});
    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[4], positions[5], radius});
    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[6], positions[7], radius});

    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[0], positions[2], radius});
    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[1], positions[3], radius});
    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[4], positions[6], radius});
    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[5], positions[7], radius});

    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[0], positions[4], radius});
    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[1], positions[5], radius});
    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[2], positions[6], radius});
    addCylinder(BOUNDINGBOX_MATERIAL_ID, {positions[3], positions[7], radius});
}

MaterialPtr OptiXModel::createMaterialImpl(const PropertyMap& properties)
{
    auto material = std::make_shared<OptiXMaterial>();
    if (!material)
        CORE_THROW(std::runtime_error("Failed to create material"));
    return material;
}

SharedDataVolumePtr OptiXModel::createSharedDataVolume(const Vector3ui& dimensions, const Vector3f& spacing,
                                                       const DataType type)
{
    if (!_geometries->_volumes.empty())
        return nullptr;

    if (!_volumeGeometries.empty())
        CORE_THROW("Only one volume per model is currently supported");

    const auto materialId = _volumeGeometries.size();
    auto material = createMaterial(materialId, "volume" + std::to_string(materialId));
    _materials[materialId] = material;

    auto volume = std::make_shared<OptiXVolume>(this, dimensions, spacing, type, _volumeParameters);
    _geometries->_volumes[materialId] = volume;

    VolumeGeometry volumeGeometry;
    volumeGeometry.dimensions = volume->getDimensions();
    volumeGeometry.offset = volume->getOffset();
    volumeGeometry.spacing = volume->getElementSpacing();
    _volumeGeometries[materialId] = volumeGeometry;

    _volumesDirty = true;
    return volume;
}

void OptiXModel::_commitVolumes(const size_t materialId)
{
    if (!_volumesDirty)
        return;

    auto iter = _geometries->_volumes.find(materialId);
    if (iter == _geometries->_volumes.end())
        return;

    _optixVolumes[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::volume);

    auto material = dynamic_cast<OptiXMaterial*>(_materials[materialId].get());
    auto optixMaterial = material->getOptixMaterial();
    if (!optixMaterial)
        CORE_THROW(std::runtime_error("OptiX material is not defined"));

    auto& optixVolumes = _optixVolumes[materialId];
    optixVolumes->setPrimitiveCount(1);

    auto context = OptiXContext::get().getOptixContext();
    auto instance = context->createGeometryInstance();
    auto& optixVolume = _optixVolumes[materialId];
    instance->setGeometry(optixVolume);
    instance->setMaterialCount(1);
    instance->setMaterial(0, optixMaterial);
    _geometryGroup->addChild(instance);

    const auto volume = dynamic_cast<OptiXVolume*>((*iter).second.get());
    _volumeGeometries[materialId].offset = volume->getOffset();
    _volumeGeometries[materialId].octreeDataType = volume->getOctreeDataType();

    const auto& memoryBuffer = volume->getMemoryBuffer();
    if (!memoryBuffer.empty())
    {
        // Volume as 3D texture
        const auto& dimensions = volume->getDimensions();
        const auto& valueRange = volume->getValueRange();
        Buffer buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, dimensions.x, dimensions.y,
                                                       dimensions.z, 1u);
        const size_t size = dimensions.x * dimensions.y * dimensions.z;
        memcpy(buffer->map(), memoryBuffer.data(), size * sizeof(float));
        buffer->unmap();
        _createSampler(materialId, buffer, size, TextureType::volume, RT_TEXTURE_INDEX_ARRAY_INDEX, valueRange);
    }

    const auto& octreeIndices = volume->getOctreeIndices();
    if (!octreeIndices.empty())
    {
        // Octree indices as texture
        const size_t size = octreeIndices.size();
        Buffer buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, size, 1u);
        memcpy(buffer->map(), octreeIndices.data(), size * sizeof(uint32_t));
        buffer->unmap();
        _createSampler(materialId, buffer, size, TextureType::octree_indices, RT_TEXTURE_INDEX_ARRAY_INDEX);
    }

    const auto& octreeValues = volume->getOctreeValues();
    if (!octreeValues.empty())
    {
        // Octree values as texture
        const size_t size = octreeValues.size();
        Buffer buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, size, 1u);
        memcpy(buffer->map(), octreeValues.data(), size * sizeof(float));
        buffer->unmap();
        _createSampler(materialId, buffer, size, TextureType::octree_values, RT_TEXTURE_INDEX_ARRAY_INDEX);
    }
    _commitVolumesBuffers(materialId);
    _volumesDirty = false;
}

void OptiXModel::_commitVolumesBuffers(const size_t materialId)
{
    if (_volumeGeometries.empty())
        return;

    std::vector<VolumeGeometry> volumeGeometries;
    for (const auto& volumeGeometry : _volumeGeometries)
        volumeGeometries.push_back(volumeGeometry.second);

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _volumesBuffers[materialId],
              _optixVolumes[materialId][OPTIX_GEOMETRY_PROPERTY_VOLUMES], volumeGeometries,
              sizeof(VolumeGeometry) * volumeGeometries.size());
}

BrickedVolumePtr OptiXModel::createBrickedVolume(const Vector3ui& /*dimensions*/, const Vector3f& /*spacing*/,
                                                 const DataType /*type*/)
{
    throw std::runtime_error("Not implemented");
    return nullptr;
}

void OptiXModel::_commitTransferFunctionImpl(const Vector3fs& colors, const floats& opacities,
                                             const Vector2d valueRange)
{
    auto context = OptiXContext::get().getOptixContext();
    const auto nbColors = colors.size();
    Vector4fs colormap;
    for (uint64_t i = 0; i < nbColors; ++i)
        colormap.push_back({colors[i].x, colors[i].y, colors[i].z, opacities[i * 256 / nbColors]});

    Buffer buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nbColors, 1u);
    memcpy(buffer->map(), colormap.data(), sizeof(Vector4f) * colormap.size());
    buffer->unmap();

    // TODO: Use createSampler function!!!

    TextureSampler sampler = context->createTextureSampler();
    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    sampler->setBuffer(0u, 0u, buffer);
    sampler->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);
    const auto samplerId = sampler->getId();

    // Attach transfer function texture to all materials in the model
    for (auto& volumeGeometry : _volumeGeometries)
    {
        volumeGeometry.second.transferFunctionSamplerId = samplerId;
        volumeGeometry.second.valueRange = valueRange;
    }

    for (auto material : _materials)
    {
        const auto materialId = material.first;
        if (materialId == BOUNDINGBOX_MATERIAL_ID || materialId == SECONDARY_MODEL_MATERIAL_ID)
            continue;

        auto optixMaterial = static_cast<OptiXMaterial*>(getMaterial(materialId).get());
        auto deviceMaterial = optixMaterial->getOptixMaterial();
        if (!deviceMaterial)
            continue;

        auto& textureSamplers = optixMaterial->getTextureSamplers();
        textureSamplers.insert(std::make_pair(TextureType::transfer_function, sampler));
        const auto textureName = textureTypeToString[static_cast<uint8_t>(TextureType::transfer_function)];
        deviceMaterial[textureName]->setInt(samplerId);
        optixMaterial->setValueRange(valueRange);
        optixMaterial->commit();

        // Update volume buffers with transfer function texture sampler Id and range
        _commitVolumesBuffers(material.first);
    }
}

void OptiXModel::_commitSimulationDataImpl(const float* frameData, const size_t frameSize)
{
    auto context = OptiXContext::get().getOptixContext();
    setBufferRaw(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _userData, context[CONTEXT_USER_DATA], frameData, frameSize,
                 frameSize * sizeof(float));
}

void OptiXModel::_createSampler(const size_t materialId, const Buffer& buffer, const size_t size,
                                const TextureType textureType, const RTtextureindexmode textureIndexType,
                                const Vector2f& valueRange)
{
    auto context = OptiXContext::get().getOptixContext();
    auto material = static_cast<OptiXMaterial*>(getMaterial(materialId).get());
    auto optixMaterial = material->getOptixMaterial();
    material->setValueRange(valueRange);
    auto& textureSamplers = material->getTextureSamplers();

    // Remove existing sampler (if applicable)
    const auto it = textureSamplers.find(textureType);
    if (it != textureSamplers.end())
        textureSamplers.erase(it);

    // Create new sample
    TextureSampler sampler = context->createTextureSampler();
    const auto samplerId = sampler->getId();
    auto filteringMode = RT_FILTER_LINEAR;
    switch (textureType)
    {
    case TextureType::volume:
        _volumeGeometries[materialId].volumeSamplerId = samplerId;
        break;
    case TextureType::transfer_function:
        _volumeGeometries[materialId].transferFunctionSamplerId = samplerId;
        _volumeGeometries[materialId].valueRange = valueRange;
        break;
    case TextureType::octree_indices:
        _volumeGeometries[materialId].octreeIndicesSamplerId = samplerId;
        filteringMode = RT_FILTER_NEAREST;
        break;
    case TextureType::octree_values:
        _volumeGeometries[materialId].octreeValuesSamplerId = samplerId;
        filteringMode = RT_FILTER_NEAREST;
        break;
    }

    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setIndexingMode(textureIndexType);
    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    sampler->setBuffer(0u, 0u, buffer);
    sampler->setFilteringModes(filteringMode, filteringMode, RT_FILTER_NONE);
    sampler->setMaxAnisotropy(8.0f);
    sampler->validate();

    textureSamplers.insert(std::make_pair(textureType, sampler));
    const auto textureName = textureTypeToString[static_cast<uint8_t>(textureType)];
    optixMaterial[textureName]->setInt(samplerId);
    material->commit();
}

} // namespace optix
} // namespace engine
} // namespace core