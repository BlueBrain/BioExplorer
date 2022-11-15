/* Copyright (c) 2015-2018, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#include "OptiXModel.h"
#include "Logs.h"
#include "OptiXContext.h"
#include "OptiXMaterial.h"

#include <brayns/common/simulation/AbstractSimulationHandler.h>
#include <brayns/parameters/AnimationParameters.h>

#include <brayns/engineapi/Material.h>

#include <Exception.h>
#include <sutil.h>

namespace brayns
{
inline OptixAabb sphereBound(const Vector3f& center, const float radius)
{
    const Vector3f r{radius, radius, radius};
    const Vector3f m = center - r;
    const Vector3f M = center + r;
    return {m.x, m.y, m.z, M.x, M.y, M.z};
}

OptiXModel::OptiXModel(AnimationParameters& animationParameters,
                       VolumeParameters& volumeParameters)
    : Model(animationParameters, volumeParameters)
{
}

void OptiXModel::commitGeometry()
{
    // Materials
    _commitMaterials();

#if 0
    const auto compactBVH = getBVHFlags().count(BVHFlag::compact) > 0;
    // Geometry group
    if (!_geometryGroup)
        _geometryGroup = OptiXContext::get().createGeometryGroup(compactBVH);

    // Bounding box group
    if (!_boundingBoxGroup)
        _boundingBoxGroup = OptiXContext::get().createGeometryGroup(compactBVH);
#endif

    size_t nbSpheres = 0;
    size_t nbCylinders = 0;
    size_t nbCones = 0;
    if (_spheresDirty)
    {
        for (const auto& spheres : _geometries->_spheres)
        {
            nbSpheres += spheres.second.size();
            _commitSpheres(spheres.first);
            _createSpheresSBT(spheres.first);
            break; // TO REMOVE
        }
        PLUGIN_DEBUG(nbSpheres << " spheres");
    }

    if (_cylindersDirty)
    {
        for (const auto& cylinders : _geometries->_cylinders)
        {
            nbCylinders += cylinders.second.size();
            _commitCylinders(cylinders.first);
        }
        PLUGIN_DEBUG(nbCylinders << " cylinders");
    }

    if (_conesDirty)
    {
        for (const auto& cones : _geometries->_cones)
        {
            nbCones += cones.second.size();
            _commitCones(cones.first);
        }
        PLUGIN_DEBUG(nbCones << " cones");
    }

    if (_triangleMeshesDirty)
        for (const auto& meshes : _geometries->_triangleMeshes)
            _commitMeshes(meshes.first);

    updateBounds();
    _markGeometriesClean();

    // handled by the scene
    _instancesDirty = false;

#if 0
    PLUGIN_DEBUG("Geometry group has " << _geometryGroup->getChildCount()
                 << " children instances" );
    PLUGIN_DEBUG("Bounding box group has "
                 << _boundingBoxGroup->getChildCount() << " children instances"
                 );
#endif
}

void OptiXModel::_buildGAS(const OptixBuildInput& buildInput)
{
    PLUGIN_INFO("Registering OptiX Geometry Acceleration Structures (GAS)");
    auto& state = OptiXContext::getInstance().getState();

    OptixAccelBuildOptions accelOptions = {OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
                                           OPTIX_BUILD_OPERATION_BUILD};
    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accelOptions,
                                             &buildInput, 1, &gasBufferSizes));
    CUdeviceptr dTempBufferGas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTempBufferGas),
                          gasBufferSizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr dBufferTempOutputGasAndCompactedSize;
    size_t compactedSizeOffset =
        roundUp<size_t>(gasBufferSizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(
                              &dBufferTempOutputGasAndCompactedSize),
                          compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)(
        (char*)dBufferTempOutputGasAndCompactedSize + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(state.context,
                                0, // CUDA stream
                                &accelOptions, &buildInput,
                                1, // num build inputs
                                dTempBufferGas, gasBufferSizes.tempSizeInBytes,
                                dBufferTempOutputGasAndCompactedSize,
                                gasBufferSizes.outputSizeInBytes,
                                &state.gas_handle,
                                &emitProperty, // emitted property list
                                1              // num emitted properties
                                ));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result,
                          sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gasBufferSizes.outputSizeInBytes)
    {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer),
                       compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, state.gas_handle,
                                      state.d_gas_output_buffer,
                                      compacted_gas_size, &state.gas_handle));

        CUDA_CHECK(cudaFree((void*)dBufferTempOutputGasAndCompactedSize));
    }
    else
        state.d_gas_output_buffer = dBufferTempOutputGasAndCompactedSize;
}

void OptiXModel::_commitSpheres(const size_t materialId)
{
    if (_geometries->_spheres.find(materialId) == _geometries->_spheres.end())
        return;

    auto& state = OptiXContext::getInstance().getState();
    const auto& spheres = _geometries->_spheres[materialId];
    const auto nbSpheres = spheres.size();
    PLUGIN_INFO("Committing " << nbSpheres << " spheres");
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));

    std::vector<OptixAabb> aabb;
    aabb.resize(nbSpheres);
    uint32_ts aabbInputFlags;
    aabbInputFlags.resize(nbSpheres);
    uint32_ts sbtIndices;
    sbtIndices.resize(nbSpheres);
    uint32_t i = 0;
    for (const auto& sphere : spheres)
    {
        aabb[i] = sphereBound(sphere.center, sphere.radius);
        aabbInputFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        sbtIndices[i] = i;
        ++i;
    }

    // AABB build input
    CUdeviceptr dAabb;
    uint64_t size = nbSpheres * sizeof(OptixAabb);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dAabb), size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dAabb), aabb.data(), size,
                          cudaMemcpyHostToDevice));

    // SBT indices
    CUdeviceptr dSbtIndices;
    size = nbSpheres * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSbtIndices), size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSbtIndices),
                          sbtIndices.data(), size, cudaMemcpyHostToDevice));

    // AABB description
    OptixBuildInput aabbInput = {};
    aabbInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabbInput.customPrimitiveArray.aabbBuffers = &dAabb;
    aabbInput.customPrimitiveArray.flags = aabbInputFlags.data();
    aabbInput.customPrimitiveArray.numSbtRecords = nbSpheres;
    aabbInput.customPrimitiveArray.numPrimitives = nbSpheres;
    aabbInput.customPrimitiveArray.sbtIndexOffsetBuffer = dSbtIndices;
    aabbInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    aabbInput.customPrimitiveArray.primitiveIndexOffset = 0;

    _buildGAS(aabbInput);

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dSbtIndices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dAabb)));
}

void OptiXModel::_createSpheresSBT(const size_t materialId)
{
    auto& state = OptiXContext::getInstance().getState();
    const auto& spheres = _geometries->_spheres[materialId];
    const auto nbSpheres = spheres.size();
    PLUGIN_DEBUG("Creating OptiX SBT group records");
    const size_t nbRecords = RAY_TYPE_COUNT * nbSpheres;
    std::vector<HitGroupRecord> hitgroupRecords;
    hitgroupRecords.resize(nbRecords);

    size_t index = 0;
    for (const auto& sphere : spheres)
    {
        // Radiance
        const auto& center = sphere.center;
        GeometryData::Sphere optixSphere{{center.x, center.y, center.z},
                                         sphere.radius};
        OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_prog_group,
                                             &hitgroupRecords[index]));
        hitgroupRecords[index].data.geometry.sphere = optixSphere;
        hitgroupRecords[index].data.shading.phong = {
            {0.f, 0.f, 0.f},                                              // Ka
            {rand() % 10 / 10.f, rand() % 10 / 10.f, rand() % 10 / 10.f}, // Kd
            {0.9f, 0.9f, 0.9f},                                           // Ks
            {0.f, 0.f, 0.f},                                              // Kr
            128, // phong_exp
        };
        ++index;

        // Occlusion
        OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_prog_group,
                                             &hitgroupRecords[index]));
        hitgroupRecords[index].data.geometry.sphere = optixSphere;
        ++index;
    }

    PLUGIN_INFO("Registering " << index << "/" << nbRecords
                               << " hitgroup records");
    CUdeviceptr dHitgroupRecords;
    const size_t bufferSize = sizeof(HitGroupRecord) * nbRecords;
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&dHitgroupRecords), bufferSize));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dHitgroupRecords),
                          hitgroupRecords.data(), bufferSize,
                          cudaMemcpyHostToDevice));

    state.sbt.hitgroupRecordBase = dHitgroupRecords;
    state.sbt.hitgroupRecordCount = nbRecords;
    state.sbt.hitgroupRecordStrideInBytes =
        static_cast<uint32_t>(sizeof(HitGroupRecord));
}

void OptiXModel::_commitCylinders(const size_t materialId)
{
    if (_geometries->_cylinders.find(materialId) ==
        _geometries->_cylinders.end())
        return;
    // const auto& cylinders = _geometries->_cylinders[materialId];
}

void OptiXModel::_commitCones(const size_t materialId)
{
    if (_geometries->_cones.find(materialId) == _geometries->_cones.end())
        return;

    // const auto& cones = _geometries->_cones[materialId];
}

void OptiXModel::_commitMeshes(const size_t materialId)
{
    if (_geometries->_triangleMeshes.find(materialId) ==
        _geometries->_triangleMeshes.end())
        return;

    // const auto& meshes = _geometries->_triangleMeshes[materialId];
}

void OptiXModel::_commitMaterials()
{
    PLUGIN_INFO("Committing " << _materials.size() << " OptiX materials");

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
        {c.x - s.x, c.y - s.y, c.z - s.z},
        {c.x + s.x, c.y - s.y, c.z - s.z}, //    6--------7
        {c.x - s.x, c.y + s.y, c.z - s.z}, //   /|       /|
        {c.x + s.x, c.y + s.y, c.z - s.z}, //  2--------3 |
        {c.x - s.x, c.y - s.y, c.z + s.z}, //  | |      | |
        {c.x + s.x, c.y - s.y, c.z + s.z}, //  | 4------|-5
        {c.x - s.x, c.y + s.y, c.z + s.z}, //  |/       |/
        {c.x + s.x, c.y + s.y, c.z + s.z}  //  0--------1
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

MaterialPtr OptiXModel::createMaterialImpl(
    const PropertyMap& properties BRAYNS_UNUSED)
{
    auto material = std::make_shared<OptiXMaterial>();
    if (!material)
        BRAYNS_THROW(std::runtime_error("Failed to create material"));
    return material;
}

SharedDataVolumePtr OptiXModel::createSharedDataVolume(
    const Vector3ui& /*dimensions*/, const Vector3f& /*spacing*/,
    const DataType /*type*/) const
{
    throw std::runtime_error("Not implemented");
    return nullptr;
}

BrickedVolumePtr OptiXModel::createBrickedVolume(
    const Vector3ui& /*dimensions*/, const Vector3f& /*spacing*/,
    const DataType /*type*/) const
{
    throw std::runtime_error("Not implemented");
    return nullptr;
}

void OptiXModel::_commitTransferFunctionImpl(const Vector3fs& colors,
                                             const floats& opacities,
                                             const Vector2d valueRange)
{
}

void OptiXModel::_commitSimulationDataImpl(const float* frameData,
                                           const size_t frameSize)
{
}
} // namespace brayns
