/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
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
inline OptixAabb sphereBounds(const Vector3f& center, const float radius)
{
    const Vector3f r{radius, radius, radius};
    const Vector3f m = center - r;
    const Vector3f M = center + r;
    return {m.x, m.y, m.z, M.x, M.y, M.z};
}

inline OptixAabb cylinderBounds(const Vector3f& center, const Vector3f& up, const float radius)
{
    const Vector3f m = {std::min(center.x, up.x) - radius, std::min(center.y, up.y) - radius,
                        std::min(center.z, up.z) - radius};
    const Vector3f M = {std::max(center.x, up.x) + radius, std::max(center.y, up.y) + radius,
                        std::max(center.z, up.z) + radius};
    return {m.x, m.y, m.z, M.x, M.y, M.z};
}

inline OptixAabb coneBounds(const Vector3f& center, const Vector3f& up, const float centerRadius, const float upRadius)
{
    const float radius = std::max(centerRadius, upRadius);
    const Vector3f m = {std::min(center.x, up.x) - radius, std::min(center.y, up.y) - radius,
                        std::min(center.z, up.z) - radius};
    const Vector3f M = {std::max(center.x, up.x) + radius, std::max(center.y, up.y) + radius,
                        std::max(center.z, up.z) + radius};
    return {m.x, m.y, m.z, M.x, M.y, M.z};
}

OptiXModel::OptiXModel(AnimationParameters& animationParameters, VolumeParameters& volumeParameters)
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

    if (_spheresDirty || _cylindersDirty || _conesDirty)
    {
        _commitGeometry();
        _createSBT();
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

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));

    OptixAccelBuildOptions accelOptions = {OPTIX_BUILD_FLAG_ALLOW_COMPACTION, OPTIX_BUILD_OPERATION_BUILD};
    OptixAccelBufferSizes gasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accelOptions, &buildInput, 1, &gasBufferSizes));
    CUdeviceptr dTempBufferGas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTempBufferGas), gasBufferSizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr dBufferTempOutputGasAndCompactedSize;
    size_t compactedSizeOffset = roundUp<size_t>(gasBufferSizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dBufferTempOutputGasAndCompactedSize), compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)dBufferTempOutputGasAndCompactedSize + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(state.context,
                                0, // CUDA stream
                                &accelOptions, &buildInput,
                                1, // num build inputs
                                dTempBufferGas, gasBufferSizes.tempSizeInBytes, dBufferTempOutputGasAndCompactedSize,
                                gasBufferSizes.outputSizeInBytes, &state.gas_handle,
                                &emitProperty, // emitted property list
                                1              // num emitted properties
                                ));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gasBufferSizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));
        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size,
                                      &state.gas_handle));

        CUDA_CHECK(cudaFree((void*)dBufferTempOutputGasAndCompactedSize));
    }
    else
        state.d_gas_output_buffer = dBufferTempOutputGasAndCompactedSize;
}

void OptiXModel::_commitGeometry()
{
    std::vector<OptixAabb> optixAABBs;

    for (const auto& geometries : _geometries->_spheres)
        for (const auto& geometry : geometries.second)
            optixAABBs.push_back(sphereBounds(geometry.center, geometry.radius));

    for (const auto& geometries : _geometries->_cylinders)
        for (const auto& geometry : geometries.second)
            optixAABBs.push_back(cylinderBounds(geometry.center, geometry.up, geometry.radius));

    for (const auto& geometries : _geometries->_cones)
        for (const auto& geometry : geometries.second)
            optixAABBs.push_back(coneBounds(geometry.center, geometry.up, geometry.centerRadius, geometry.upRadius));

    const auto nbAABBs = optixAABBs.size();
    PLUGIN_INFO("Number of AABB: " << nbAABBs);

    uint32_ts sbtIndices;
    uint32_ts aabbInputFlags;
    sbtIndices.resize(nbAABBs);
    aabbInputFlags.resize(nbAABBs);
    for (uint32_t i = 0; i < nbAABBs; ++i)
    {
        sbtIndices[i] = i;
        aabbInputFlags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    }

    // AABB build input
    CUdeviceptr dAABBs = 0;
    uint64_t size = nbAABBs * sizeof(OptixAabb);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dAABBs), size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dAABBs), optixAABBs.data(), size, cudaMemcpyHostToDevice));

    // SBT indices
    CUdeviceptr dSBTIndices = 0;
    size = nbAABBs * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSBTIndices), size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSBTIndices), sbtIndices.data(), size, cudaMemcpyHostToDevice));

    // AABB description
    OptixBuildInput aabbInput = {};
    aabbInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabbInput.customPrimitiveArray.aabbBuffers = &dAABBs;
    aabbInput.customPrimitiveArray.flags = aabbInputFlags.data();
    aabbInput.customPrimitiveArray.numSbtRecords = nbAABBs;
    aabbInput.customPrimitiveArray.numPrimitives = nbAABBs;
    aabbInput.customPrimitiveArray.sbtIndexOffsetBuffer = dSBTIndices;
    aabbInput.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    aabbInput.customPrimitiveArray.primitiveIndexOffset = 0;

    _buildGAS(aabbInput);

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dAABBs)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dSBTIndices)));
}

void OptiXModel::_createSBT()
{
    auto& state = OptiXContext::getInstance().getState();
    PLUGIN_DEBUG("Creating OptiX SBT Group records");
    std::vector<HitGroupRecord> hitGroupRecords;

    // Spheres
    for (const auto& geometries : _geometries->_spheres)
    {
        const auto material = _materials[geometries.first];
        const Vector3f kd = material->getDiffuseColor();
        const Vector3f ks = material->getSpecularColor();
        const float r = material->getReflectionIndex();
        const float phongExp = material->getSpecularExponent();
        for (const auto& geometry : geometries.second)
        {
            const auto& center = geometry.center;
            const GeometryData::Sphere optixGeometry{{center.x, center.y, center.z}, geometry.radius};
            {
                // Radiance
                HitGroupRecord hitGroupRecord;
                OPTIX_CHECK(optixSbtRecordPackHeader(state.sphere_radiance_prog_group, &hitGroupRecord));
                hitGroupRecord.data.geometry.sphere = optixGeometry;
                hitGroupRecord.data.shading.phong = {
                    {0.f, 0.f, 0.f},    // Ka
                    {kd.x, kd.y, kd.z}, // Kd
                    {ks.x, ks.y, ks.z}, // Ks
                    {r, r, r},          // Kr
                    phongExp,           // phong_exp
                };
                hitGroupRecords.push_back(hitGroupRecord);
            }

            {
                // Occlusion
                HitGroupRecord hitGroupRecord;
                OPTIX_CHECK(optixSbtRecordPackHeader(state.sphere_occlusion_prog_group, &hitGroupRecord));
                hitGroupRecord.data.geometry.sphere = optixGeometry;
                hitGroupRecords.push_back(hitGroupRecord);
            }
        }
    }

    // Cylinders
    for (const auto& geometries : _geometries->_cylinders)
    {
        const auto material = _materials[geometries.first];
        const Vector3f kd = material->getDiffuseColor();
        const Vector3f ks = material->getSpecularColor();
        const float r = material->getReflectionIndex();
        const float phongExp = material->getSpecularExponent();
        for (const auto& geometry : geometries.second)
        {
            const auto& center = geometry.center;
            const auto& up = geometry.up;
            const GeometryData::Cylinder optixGeometry{{center.x, center.y, center.z},
                                                       {up.x, up.y, up.z},
                                                       geometry.radius};
            {
                // Radiance
                HitGroupRecord hitGroupRecord;
                OPTIX_CHECK(optixSbtRecordPackHeader(state.cylinder_radiance_prog_group, &hitGroupRecord));
                hitGroupRecord.data.geometry.cylinder = optixGeometry;
                hitGroupRecord.data.shading.phong = {
                    {0.f, 0.f, 0.f},    // Ka
                    {kd.x, kd.y, kd.z}, // Kd
                    {ks.x, ks.y, ks.z}, // Ks
                    {r, r, r},          // Kr
                    phongExp,           // phong_exp
                };
                hitGroupRecords.push_back(hitGroupRecord);
            }

            {
                // Occlusion
                HitGroupRecord hitGroupRecord;
                OPTIX_CHECK(optixSbtRecordPackHeader(state.cylinder_occlusion_prog_group, &hitGroupRecord));
                hitGroupRecord.data.geometry.cylinder = optixGeometry;
                hitGroupRecords.push_back(hitGroupRecord);
            }
        }
    }

    // Cones
    for (const auto& geometries : _geometries->_cones)
    {
        const auto material = _materials[geometries.first];
        const Vector3f kd = material->getDiffuseColor();
        const Vector3f ks = material->getSpecularColor();
        const float r = material->getReflectionIndex();
        const float phongExp = material->getSpecularExponent();
        for (const auto& geometry : geometries.second)
        {
            const auto& center = geometry.center;
            const auto& up = geometry.up;
            const auto centerRadius = geometry.centerRadius;
            const auto upRadius = geometry.upRadius;
            const GeometryData::Cone optixGeometry{{center.x, center.y, center.z},
                                                   {up.x, up.y, up.z},
                                                   centerRadius,
                                                   upRadius};
            {
                // Radiance
                HitGroupRecord hitGroupRecord;
                OPTIX_CHECK(optixSbtRecordPackHeader(state.cone_radiance_prog_group, &hitGroupRecord));
                hitGroupRecord.data.geometry.cone = optixGeometry;
                hitGroupRecord.data.shading.phong = {
                    {0.f, 0.f, 0.f},    // Ka
                    {kd.x, kd.y, kd.z}, // Kd
                    {ks.x, ks.y, ks.z}, // Ks
                    {r, r, r},          // Kr
                    phongExp,           // phong_exp
                };
                hitGroupRecords.push_back(hitGroupRecord);
            }

            {
                // Occlusion
                HitGroupRecord hitGroupRecord;
                OPTIX_CHECK(optixSbtRecordPackHeader(state.cone_occlusion_prog_group, &hitGroupRecord));
                hitGroupRecord.data.geometry.cone = optixGeometry;
                hitGroupRecords.push_back(hitGroupRecord);
            }
        }
    }

    PLUGIN_INFO("Registering " << hitGroupRecords.size() << " Hit Group Records");
    CUdeviceptr dHitGroupRecords = 0;
    const uint32_t hitGroupRecordSize = sizeof(HitGroupRecord);
    PLUGIN_DEBUG("HitGroupData          : " << sizeof(HitGroupData));
    PLUGIN_DEBUG("HitGroupData(geometry): " << sizeof(HitGroupData::geometry));
    PLUGIN_DEBUG("HitGroupData(shading) : " << sizeof(HitGroupData::shading));
    PLUGIN_DEBUG("HitGroupData(total)   : " << sizeof(HitGroupData) + OPTIX_SBT_RECORD_HEADER_SIZE);
    PLUGIN_DEBUG("HitGroupRecord        : " << hitGroupRecordSize);

    const size_t bufferSize = hitGroupRecordSize * hitGroupRecords.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dHitGroupRecords), bufferSize));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dHitGroupRecords), hitGroupRecords.data(), bufferSize,
                          cudaMemcpyHostToDevice));

    // CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));

    state.sbt.hitgroupRecordBase = dHitGroupRecords;
    state.sbt.hitgroupRecordCount = hitGroupRecords.size();
    state.sbt.hitgroupRecordStrideInBytes = hitGroupRecordSize;
}

void OptiXModel::_commitMeshes(const size_t materialId)
{
    if (_geometries->_triangleMeshes.find(materialId) == _geometries->_triangleMeshes.end())
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

MaterialPtr OptiXModel::createMaterialImpl(const PropertyMap& properties BRAYNS_UNUSED)
{
    auto material = std::make_shared<OptiXMaterial>();
    if (!material)
        BRAYNS_THROW(std::runtime_error("Failed to create material"));
    return material;
}

SharedDataVolumePtr OptiXModel::createSharedDataVolume(const Vector3ui& /*dimensions*/, const Vector3f& /*spacing*/,
                                                       const DataType /*type*/) const
{
    throw std::runtime_error("Not implemented");
    return nullptr;
}

BrickedVolumePtr OptiXModel::createBrickedVolume(const Vector3ui& /*dimensions*/, const Vector3f& /*spacing*/,
                                                 const DataType /*type*/) const
{
    throw std::runtime_error("Not implemented");
    return nullptr;
}

void OptiXModel::_commitTransferFunctionImpl(const Vector3fs& colors, const floats& opacities,
                                             const Vector2d valueRange)
{
}

void OptiXModel::_commitSimulationDataImpl(const float* frameData, const size_t frameSize) {}
} // namespace brayns
