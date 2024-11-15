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

#include "OptiXModel.h"
#include "OptiXCommonStructs.h"
#include "OptiXContext.h"
#include "OptiXField.h"
#include "OptiXMaterial.h"
#include "OptiXUtils.h"
#include "OptiXVolume.h"

#include <platform/core/common/Logs.h>
#include <platform/core/common/simulation/AbstractAnimationHandler.h>
#include <platform/core/common/utils/Utils.h>
#include <platform/core/parameters/AnimationParameters.h>

#include <platform/core/engineapi/Material.h>

#include <omp.h>

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
                       GeometryParameters& geometryParameters, FieldParameters& fieldParameters)
    : Model(animationParameters, volumeParameters, geometryParameters, fieldParameters)
{
}

OptiXModel::~OptiXModel()
{
    RT_DESTROY_MAP(_optixSpheres);
    RT_DESTROY_MAP(_spheresBuffers);

    RT_DESTROY_MAP(_optixCylinders)
    RT_DESTROY_MAP(_cylindersBuffers);

    RT_DESTROY_MAP(_optixCones)
    RT_DESTROY_MAP(_conesBuffers);

    for (auto sdfGeometriesBuffers : _sdfGeometriesBuffers)
    {
        RT_DESTROY(sdfGeometriesBuffers.second.geometries_buffer);
        RT_DESTROY(sdfGeometriesBuffers.second.neighbours_buffer);
        RT_DESTROY(sdfGeometriesBuffers.second.indices_buffer);
    }

    RT_DESTROY_MAP(_optixVolumes)
    RT_DESTROY_MAP(_volumesBuffers);

    RT_DESTROY_MAP(_optixFields)
    RT_DESTROY_MAP(_fieldsBuffers);

    RT_DESTROY_MAP(_optixMeshes)
    for (auto optixMeshBuffers : _optixMeshBuffers)
    {
        RT_DESTROY(optixMeshBuffers.second.vertices_buffer);
        RT_DESTROY(optixMeshBuffers.second.normal_buffer);
        RT_DESTROY(optixMeshBuffers.second.texcoord_buffer);
        RT_DESTROY(optixMeshBuffers.second.indices_buffer);
    }

    RT_DESTROY_MAP(_optixStreamlines)
    for (auto streamlinesBuffers : _streamlinesBuffers)
    {
        RT_DESTROY(streamlinesBuffers.second.vertices_buffer);
        RT_DESTROY(streamlinesBuffers.second.color_buffer);
        RT_DESTROY(streamlinesBuffers.second.indices_buffer);
    }

    RT_DESTROY_MAP(_optixTextures)
    RT_DESTROY_MAP(_optixTextureSamplers)
    // TODO: Each geometry should have its own userDataBuffer. This is something to be handled by the handler attached
    // to the model
    // _resetUserDataBuffer();

    RT_DESTROY(_geometryGroup);
    RT_DESTROY(_boundingBoxGroup);
}

void OptiXModel::commitGeometry()
{
    // Materials
    uint64_t memoryFootPrint = _commitMaterials();

    if (!isDirty())
        return;

    const auto compactBVH = getBVHFlags().count(BVHFlag::compact) > 0;
    auto& context = OptiXContext::get();
    // Geometry group
    if (!_geometryGroup)
        _geometryGroup = context.createGeometryGroup(compactBVH);

    // Bounding box group
    if (!_boundingBoxGroup)
        _boundingBoxGroup = context.createGeometryGroup(compactBVH);

    // Geometry
    updateBounds();
    if (_spheresDirty)
        for (const auto& spheres : _geometries->_spheres)
            memoryFootPrint += _commitSpheres(spheres.first);

    if (_cylindersDirty)
        for (const auto& cylinders : _geometries->_cylinders)
            memoryFootPrint += _commitCylinders(cylinders.first);

    if (_conesDirty)
        for (const auto& cones : _geometries->_cones)
            memoryFootPrint += _commitCones(cones.first);

    if (_sdfGeometriesDirty)
        memoryFootPrint += _commitSDFGeometries();

    if (_triangleMeshesDirty)
        for (const auto& meshes : _geometries->_triangleMeshes)
            memoryFootPrint += _commitMeshes(meshes.first);

    if (_volumesDirty)
        for (const auto& volume : _geometries->_volumes)
            memoryFootPrint += _commitVolumes(volume.first);

    if (_streamlinesDirty)
        for (const auto& streamlines : _geometries->_streamlines)
            memoryFootPrint += _commitStreamlines(streamlines.first);

    if (_fieldsDirty)
        for (const auto& field : _geometries->_fields)
            memoryFootPrint += _commitFields(field.first);

    _markGeometriesClean();
    _transferFunction.markModified();
    _instancesDirty = false;
    CORE_DEBUG("Geometry group has " << _geometryGroup->getChildCount() << " children instances");
    CORE_DEBUG("Bounding box group has " << _boundingBoxGroup->getChildCount() << " children instances");
    CORE_DEBUG("Model memory footprint: " << memoryFootPrint / 1024 / 1024 << " MB");
}

void OptiXModel::_resetUserDataBuffer()
{
    // Buffer needs to be bound. Initialize it to size 1 if user data is empty
    RT_DESTROY(_userDataBuffer);
    floats frameData(1, 0);
    auto context = OptiXContext::get().getOptixContext();
    setBufferRaw(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _userDataBuffer, context[CONTEXT_USER_DATA], frameData.data(),
                 frameData.size(), frameData.size() * sizeof(float));
}

uint64_t OptiXModel::_commitSpheres(const size_t materialId)
{
    uint64_t memoryFootPrint = 0;
    if (_geometries->_spheres.find(materialId) == _geometries->_spheres.end())
        return memoryFootPrint;

    auto context = OptiXContext::get().getOptixContext();
    const auto& spheres = _geometries->_spheres[materialId];
    context[CONTEXT_SPHERE_SIZE]->setUint(sizeof(Sphere) / sizeof(float));

    // Geometry
    _optixSpheres[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::sphere);
    _optixSpheres[materialId]->setPrimitiveCount(spheres.size());

    const uint64_t bufferSize = sizeof(Sphere) * spheres.size();
    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _spheresBuffers[materialId],
              _optixSpheres[materialId][OPTIX_GEOMETRY_PROPERTY_SPHERES], spheres, bufferSize);
    memoryFootPrint += bufferSize;

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
    CORE_DEBUG("Spheres memory footprint: " << memoryFootPrint / 1024 / 1024 << " MB");
    return memoryFootPrint;
}

uint64_t OptiXModel::_commitCylinders(const size_t materialId)
{
    uint64_t memoryFootPrint = 0;
    if (_geometries->_cylinders.find(materialId) == _geometries->_cylinders.end())
        return memoryFootPrint;

    auto context = OptiXContext::get().getOptixContext();
    const auto& cylinders = _geometries->_cylinders[materialId];
    context[CONTEXT_CYLINDER_SIZE]->setUint(sizeof(Cylinder) / sizeof(float));
    _optixCylinders[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::cylinder);

    auto& optixCylinders = _optixCylinders[materialId];
    optixCylinders->setPrimitiveCount(cylinders.size());

    const uint64_t bufferSize = sizeof(Cylinder) * cylinders.size();
    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _cylindersBuffers[materialId],
              _optixCylinders[materialId][OPTIX_GEOMETRY_PROPERTY_CYLINDERS], cylinders, bufferSize);
    memoryFootPrint += bufferSize;

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
    CORE_DEBUG("Cylinders memory footprint: " << memoryFootPrint / 1024 / 1024 << " MB");
    return memoryFootPrint;
}

uint64_t OptiXModel::_commitCones(const size_t materialId)
{
    uint64_t memoryFootPrint = 0;
    if (_geometries->_cones.find(materialId) == _geometries->_cones.end())
        return memoryFootPrint;

    auto context = OptiXContext::get().getOptixContext();
    const auto& cones = _geometries->_cones[materialId];
    context[CONTEXT_CONE_SIZE]->setUint(sizeof(Cone) / sizeof(float));
    _optixCones[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::cone);

    auto& optixCones = _optixCones[materialId];
    optixCones->setPrimitiveCount(cones.size());

    const uint64_t bufferSize = sizeof(Cone) * cones.size();
    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _conesBuffers[materialId],
              _optixCones[materialId][OPTIX_GEOMETRY_PROPERTY_CONES], cones, bufferSize);
    memoryFootPrint += bufferSize;

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
    CORE_DEBUG("Cones memory footprint: " << memoryFootPrint / 1024 / 1024 << " MB");
    return memoryFootPrint;
}

uint64_t OptiXModel::_commitSDFGeometries()
{
    uint64_t memoryFootPrint = 0;
    if (_geometries->_sdf.geometries.empty())
        return memoryFootPrint;

    auto context = OptiXContext::get().getOptixContext();
    auto& sdfGeometries = _geometries->_sdf;
    const uint32_t sdfGeometrySize = sizeof(SDFGeometry);
    context[CONTEXT_SDF_GEOMETRY_SIZE]->setUint(sdfGeometrySize);

    const uint64_t nbGeometries = sdfGeometries.geometries.size();
    const uint64_t geometriesBufferSize = sizeof(SDFGeometry) * nbGeometries;

    uint64_ts geometryMaterials;
    for (const auto& geometryIndex : sdfGeometries.geometryIndices)
        geometryMaterials.push_back(geometryIndex.first);
    const uint64_t nbMaterials = geometryMaterials.size();

    uint64_t i = 0;
#pragma omp parallel for private(memoryFootPrint)
    for (i = 0; i < nbMaterials; ++i)
    {
        const auto materialId = geometryMaterials[i];

        // Indices of geometries for current material id
        const auto& geometryIndices = sdfGeometries.geometryIndices[materialId];

        // Create a local copy of SDF geometries attached to the material id
        std::map<uint64_t, SDFGeometry*> localGeometries;

        for (const auto geometryIndex : geometryIndices)
        {
            localGeometries[geometryIndex] = &sdfGeometries.geometries[geometryIndex];
            const auto& neighbours = sdfGeometries.neighbours[geometryIndex];
            for (const auto neighbour : neighbours)
                localGeometries[neighbour] = &sdfGeometries.geometries[neighbour];
        }

        uint64_t index = 0;
        uint64_tm localGeometriesMapping;
        for (const auto& localGeometry : localGeometries)
        {
            localGeometriesMapping[localGeometry.first] = index;
            ++index;
        }

        uint64_ts localIndicesFlat;
        localIndicesFlat.reserve(geometryIndices.size());
        for (const auto geometryIndex : geometryIndices)
            localIndicesFlat.push_back(localGeometriesMapping[geometryIndex]);

        if (localIndicesFlat.empty())
            continue;

        // Create a local flat representation of the SDF geometries neighbours
        uint64_ts localNeighboursFlat;
        for (const auto geometryIndex : geometryIndices)
        {
            const auto& neighbours = sdfGeometries.neighbours[geometryIndex];
            localGeometries[geometryIndex]->neighboursIndex = localNeighboursFlat.size();
            localGeometries[geometryIndex]->numNeighbours = static_cast<uint8_t>(neighbours.size());
            for (uint64_t i = 0; i < neighbours.size(); ++i)
            {
                const auto it = localGeometriesMapping.find(neighbours[i]);
                if (it == localGeometriesMapping.end())
                    CORE_THROW("Invalid neighbour index");
                if (geometryIndex != (*it).second)
                    localNeighboursFlat.push_back((*it).second);
            }
        }

        // Convert map to vector in order to send the data to the device
        std::vector<SDFGeometry> localGeometriesFlat;
        localGeometriesFlat.reserve(localGeometries.size());
        for (const auto geometry : localGeometries)
            localGeometriesFlat.push_back(*geometry.second);

        // Prepare OptiX geometry and corresponding buffers
        _optixSdfGeometries[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::sdfGeometry);
        _optixSdfGeometries[materialId]->setPrimitiveCount(localIndicesFlat.size());

        uint64_t bufferSize = localIndicesFlat.size();
#pragma omp critical
        setBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_LONG_LONG, _sdfGeometriesBuffers[materialId].indices_buffer,
                  _optixSdfGeometries[materialId][OPTIX_GEOMETRY_PROPERTY_SDF_GEOMETRIES_INDICES], localIndicesFlat,
                  bufferSize);
#pragma omp critical
        memoryFootPrint += bufferSize;

        bufferSize = localNeighboursFlat.size();
#pragma omp critical
        setBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_LONG_LONG, _sdfGeometriesBuffers[materialId].neighbours_buffer,
                  _optixSdfGeometries[materialId][OPTIX_GEOMETRY_PROPERTY_SDF_GEOMETRIES_NEIGHBOURS],
                  localNeighboursFlat, bufferSize);
#pragma omp critical
        memoryFootPrint += bufferSize;

        bufferSize = sizeof(SDFGeometry) * localGeometriesFlat.size();
#pragma omp critical
        setBuffer(RT_BUFFER_INPUT, RT_FORMAT_BYTE, _sdfGeometriesBuffers[materialId].geometries_buffer,
                  _optixSdfGeometries[materialId][OPTIX_GEOMETRY_PROPERTY_SDF_GEOMETRIES], localGeometriesFlat,
                  bufferSize);
#pragma omp critical
        memoryFootPrint += bufferSize;

        // Create material
        auto& material = static_cast<OptiXMaterial&>(*_materials[materialId]);
        auto optixMaterial = material.getOptixMaterial();
        if (!optixMaterial)
            CORE_THROW(std::runtime_error("OptiX material is not defined"));

        auto instance = context->createGeometryInstance();
        instance->setGeometry(_optixSdfGeometries[materialId]);
        instance->setMaterialCount(1);
        instance->setMaterial(0, optixMaterial);

        // Add geometry to the model
#pragma omp critical
        _geometryGroup->addChild(instance);
    }
    CORE_DEBUG("SDFGeometries memory footprint: " << memoryFootPrint / 1024 / 1024 << " MB");
    return memoryFootPrint;
}

uint64_t OptiXModel::_commitMeshes(const size_t materialId)
{
    uint64_t memoryFootPrint = 0;
    if (_geometries->_triangleMeshes.find(materialId) == _geometries->_triangleMeshes.end())
        return memoryFootPrint;

    const auto& meshes = _geometries->_triangleMeshes[materialId];
    _optixMeshes[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::triangleMesh);

    auto& optixMeshes = _optixMeshes[materialId];
    optixMeshes->setPrimitiveCount(meshes.indices.size());

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, _optixMeshBuffers[materialId].vertices_buffer,
              _optixMeshes[materialId][OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_VERTEX], meshes.vertices,
              meshes.vertices.size());
    memoryFootPrint += meshes.vertices.size() * sizeof(float) * 3;

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, _optixMeshBuffers[materialId].indices_buffer,
              _optixMeshes[materialId][OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_INDEX], meshes.indices,
              meshes.indices.size());
    memoryFootPrint += meshes.indices.size() * sizeof(uint) * 3;

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, _optixMeshBuffers[materialId].normal_buffer,
              _optixMeshes[materialId][OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_NORMAL], meshes.normals,
              meshes.normals.size());
    memoryFootPrint += meshes.normals.size() * sizeof(float) * 3;

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, _optixMeshBuffers[materialId].texcoord_buffer,
              _optixMeshes[materialId][OPTIX_GEOMETRY_PROPERTY_TRIANGLE_MESH_TEXTURE_COORDINATES],
              meshes.textureCoordinates, meshes.textureCoordinates.size());
    memoryFootPrint += meshes.textureCoordinates.size() * sizeof(float) * 2;

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
    CORE_DEBUG("Meshes memory footprint: " << memoryFootPrint / 1024 / 1024 << " MB");
    return memoryFootPrint;
}

uint64_t OptiXModel::_commitStreamlines(const size_t materialId)
{
    uint64_t memoryFootPrint = 0;
    if (_geometries->_streamlines.find(materialId) == _geometries->_streamlines.end())
        return memoryFootPrint;

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
    memoryFootPrint += vertexCurve.size() * sizeof(float) * 4;

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, _streamlinesBuffers[materialId].indices_buffer,
              _optixStreamlines[materialId][OPTIX_GEOMETRY_PROPERTY_STREAMLINE_MESH_INDEX], indexCurve,
              indexCurve.size());
    memoryFootPrint += indexCurve.size() * sizeof(uint);

    auto& material = static_cast<OptiXMaterial&>(*_materials[materialId]);
    auto optixMaterial = material.getOptixMaterial();
    if (!optixMaterial)
        CORE_THROW(std::runtime_error("OptiX material is not defined"));

    auto context = OptiXContext::get().getOptixContext();

    const Vector2ui textureSize{MAX_TEXTURE_SIZE, 1 + colorCurve.size() / MAX_TEXTURE_SIZE};
    Buffer buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, textureSize.x, textureSize.y, 1u);
    memcpy(buffer->map(), colorCurve.data(), sizeof(Vector4f) * colorCurve.size());
    buffer->unmap();
    memoryFootPrint += textureSize.x * textureSize.y * sizeof(float) * 4;

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
    CORE_DEBUG("Streamlines memory footprint: " << memoryFootPrint / 1024 / 1024 << " MB");
    return memoryFootPrint;
}

uint64_t OptiXModel::_commitMaterials()
{
    uint64_t memoryFootPrint = 0;
    CORE_DEBUG("Committing " << _materials.size() << " OptiX materials");

    for (auto& material : _materials)
        material.second->commit();
    CORE_DEBUG("Materials memory footprint: " << memoryFootPrint / 1024 / 1024 << " MB");
    return memoryFootPrint;
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

    const auto materialId = VOLUME_MATERIAL_ID;
    auto material = createMaterial(materialId, "volume" + std::to_string(materialId));
    _materials[materialId] = material;

    const auto volume = std::make_shared<OptiXSharedDataVolume>(dimensions, spacing, type, _volumeParameters);
    _geometries->_volumes[materialId] = volume;

    VolumeGeometry volumeGeometry;
    volumeGeometry.dimensions = volume->getDimensions();
    volumeGeometry.offset = volume->getOffset();
    volumeGeometry.spacing = volume->getElementSpacing();
    _volumeGeometries[materialId] = volumeGeometry;

    _volumesDirty = true;
    return volume;
}

FieldPtr OptiXModel::createField(const Vector3ui& dimensions, const Vector3f& spacing, const Vector3f& offset,
                                 const uint32_ts& indices, const floats& values, const OctreeDataType dataType)
{
    if (!_geometries->_volumes.empty())
        return nullptr;

    if (!_volumeGeometries.empty())
        CORE_THROW("Only one volume per model is currently supported");

    const auto materialId = VOLUME_OCTREE_INDICES_MATERIAL_ID;
    auto material = createMaterial(materialId, "volume" + std::to_string(materialId));
    _materials[materialId] = material;

    const auto field =
        std::make_shared<OptiXField>(_fieldParameters, dimensions, spacing, offset, indices, values, dataType);
    _geometries->_fields[materialId] = field;

    FieldGeometry fieldGeometry;
    _fieldGeometries[materialId] = fieldGeometry;

    _fieldsDirty = true;
    return field;
}

uint64_t OptiXModel::_commitVolumes(const size_t materialId)
{
    uint64_t memoryFootPrint = 0;
    if (!_volumesDirty)
        return memoryFootPrint;

    auto iter = _geometries->_volumes.find(materialId);
    if (iter == _geometries->_volumes.end())
        return memoryFootPrint;

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

    const auto sharedDataVolume = dynamic_cast<OptiXSharedDataVolume*>(volume);
    const auto& memoryBuffer = sharedDataVolume->getMemoryBuffer();
    if (!memoryBuffer.empty())
    {
        // Volume as 3D texture
        const auto& dimensions = sharedDataVolume->getDimensions();
        const auto& valueRange = sharedDataVolume->getValueRange();
        Buffer buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, dimensions.x, dimensions.y,
                                                       dimensions.z, 1u);
        const size_t size = dimensions.x * dimensions.y * dimensions.z;
        memcpy(buffer->map(), memoryBuffer.data(), size * sizeof(float));
        buffer->unmap();
        _createSampler(materialId, buffer, TextureType::volume, RT_TEXTURE_INDEX_ARRAY_INDEX, valueRange);
        memoryFootPrint += dimensions.x * dimensions.y * dimensions.z * sizeof(float);
    }

    _commitVolumesBuffers(materialId);
    _volumesDirty = false;
    CORE_DEBUG("Volumes memory footprint: " << memoryFootPrint / 1024 / 1024 << " MB");
    return memoryFootPrint;
}

uint64_t OptiXModel::_commitFields(const size_t materialId)
{
    uint64_t memoryFootPrint = 0;
    if (!_fieldsDirty)
        return memoryFootPrint;

    auto iter = _geometries->_fields.find(materialId);
    if (iter == _geometries->_fields.end())
        return memoryFootPrint;

    _optixFields[materialId] = OptiXContext::get().createGeometry(OptixGeometryType::field);

    auto material = dynamic_cast<OptiXMaterial*>(_materials[materialId].get());
    auto optixMaterial = material->getOptixMaterial();
    if (!optixMaterial)
        CORE_THROW(std::runtime_error("OptiX material is not defined"));

    auto& optixFields = _optixFields[materialId];
    optixFields->setPrimitiveCount(1);

    auto context = OptiXContext::get().getOptixContext();
    auto instance = context->createGeometryInstance();
    auto& optixField = _optixFields[materialId];
    instance->setGeometry(optixField);
    instance->setMaterialCount(1);
    instance->setMaterial(0, optixMaterial);
    _geometryGroup->addChild(instance);

    const auto field = dynamic_cast<OptiXField*>((*iter).second.get());
    const auto& octreeIndices = field->getOctreeIndices();
    const auto& octreeValues = field->getOctreeValues();
    _fieldGeometries[materialId].dimensions = field->getDimensions();
    _fieldGeometries[materialId].spacing = field->getElementSpacing();
    _fieldGeometries[materialId].offset = field->getOffset();
    const auto fieldType = field->getOctreeDataType();
    _fieldGeometries[materialId].octreeDataType = fieldType;
    _fieldGeometries[materialId].octreeNbValues =
        octreeValues.size() /
        (fieldType == OctreeDataType::odt_points ? FIELD_POINT_DATA_SIZE : FIELD_VECTOR_DATA_SIZE);

    if (!octreeIndices.empty())
    {
        // Octree indices as texture
        const size_t size = octreeIndices.size();
        Buffer buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, size, 1u);
        memcpy(buffer->map(), octreeIndices.data(), size * sizeof(uint32_t));
        buffer->unmap();
        _fieldGeometries[materialId].octreeIndicesSamplerId =
            _createSampler(materialId, buffer, TextureType::octree_indices, RT_TEXTURE_INDEX_ARRAY_INDEX);
        memoryFootPrint += size * sizeof(uint32_t);
    }

    if (!octreeValues.empty())
    {
        // Octree values as texture
        const size_t size = octreeValues.size();
        Buffer buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, size, 1u);
        memcpy(buffer->map(), octreeValues.data(), size * sizeof(float));
        buffer->unmap();
        _fieldGeometries[materialId].octreeValuesSamplerId =
            _createSampler(materialId, buffer, TextureType::octree_values, RT_TEXTURE_INDEX_ARRAY_INDEX);
        memoryFootPrint += size * sizeof(float);
    }
    _commitFieldsBuffers(materialId);
    _fieldsDirty = false;
    CORE_DEBUG("Fields memory footprint: " << memoryFootPrint / 1024 / 1024 << " MB");
    return memoryFootPrint;
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

void OptiXModel::_commitFieldsBuffers(const size_t materialId)
{
    if (_fieldGeometries.empty())
        return;

    std::vector<FieldGeometry> fieldGeometries;
    for (const auto& fieldGeometry : _fieldGeometries)
        fieldGeometries.push_back(fieldGeometry.second);

    setBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _fieldsBuffers[materialId],
              _optixFields[materialId][OPTIX_GEOMETRY_PROPERTY_FIELDS], fieldGeometries,
              sizeof(FieldGeometry) * fieldGeometries.size());
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

    // Attach transfer function texture to all materials in the model
    for (auto material : _materials)
    {
        const auto materialId = material.first;
        if (materialId == BOUNDINGBOX_MATERIAL_ID || materialId == SECONDARY_MODEL_MATERIAL_ID)
            continue;

        auto optixMaterial = static_cast<OptiXMaterial*>(getMaterial(materialId).get());
        auto deviceMaterial = optixMaterial->getOptixMaterial();
        if (!deviceMaterial)
            continue;

        Buffer buffer = context->createMipmappedBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, nbColors, 1u);
        memcpy(buffer->map(), colormap.data(), sizeof(Vector4f) * colormap.size());
        buffer->unmap();
        const auto samplerId = _createSampler(materialId, buffer, TextureType::transfer_function,
                                              RT_TEXTURE_INDEX_NORMALIZED_COORDINATES, valueRange);

        // Update volume buffers with transfer function texture sampler Id and range
        if (materialId == VOLUME_MATERIAL_ID)
        {
            for (auto& volumeGeometry : _volumeGeometries)
            {
                volumeGeometry.second.transferFunctionSamplerId = samplerId;
                volumeGeometry.second.valueRange = valueRange;
            }
            _commitVolumesBuffers(material.first);
        }

        if (materialId == FIELD_MATERIAL_ID)
        {
            for (auto& fieldGeometry : _fieldGeometries)
            {
                fieldGeometry.second.transferFunctionSamplerId = samplerId;
                fieldGeometry.second.valueRange = valueRange;
            }
            _commitFieldsBuffers(material.first);
        }
    }
}

void OptiXModel::_commitSimulationDataImpl(const float* frameData, const size_t frameSize)
{
    auto context = OptiXContext::get().getOptixContext();
    setBufferRaw(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, _userDataBuffer, context[CONTEXT_USER_DATA], frameData, frameSize,
                 frameSize * sizeof(float));
}

size_t OptiXModel::_createSampler(const size_t materialId, const Buffer& buffer, const TextureType textureType,
                                  const RTtextureindexmode textureIndexType, const Vector2f& valueRange)
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
        if (_volumeGeometries.find(materialId) != _volumeGeometries.end())
        {
            _volumeGeometries[materialId].transferFunctionSamplerId = samplerId;
            _volumeGeometries[materialId].valueRange = valueRange;
        }
        break;
    case TextureType::octree_indices:
        _fieldGeometries[materialId].octreeIndicesSamplerId = samplerId;
        filteringMode = RT_FILTER_NEAREST;
        break;
    case TextureType::octree_values:
        _fieldGeometries[materialId].octreeValuesSamplerId = samplerId;
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
    return samplerId;
}

} // namespace optix
} // namespace engine
} // namespace core