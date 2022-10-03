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
inline OptixAabb sphere_bound(const Vector3f& center, const float radius)
{
    const Vector3f m_min{center.x - radius, center.y - radius,
                         center.z - radius};
    const Vector3f m_max{center.x + radius, center.y + radius,
                         center.z + radius};

    return {m_min.x, m_min.y, m_min.z, m_max.x, m_max.y, m_max.z};
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
#if 0
            _commitSpheres(spheres.first);
#endif
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

void OptiXModel::_commitSpheres(const size_t materialId)
{
    PLUGIN_DEBUG("Registering OptiX Geometry Acceleration Structures (GAS) ");

    if (_geometries->_spheres.find(materialId) == _geometries->_spheres.end())
        return;

    auto& state = OptiXContext::getInstance().getState();

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));

    const auto& spheres = _geometries->_spheres[materialId];
    std::vector<OptixAabb> aabbs;
    uint32_ts aabb_input_flags;
    uint32_ts sbt_indices;
    uint32_t i = 0;
    for (const auto& sphere : spheres)
    {
        aabbs.push_back(sphere_bound(sphere.center, sphere.radius));
        aabb_input_flags.push_back(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
        sbt_indices.push_back(i);
        ++i;
    }
    const auto nbPrimitives = aabbs.size();

    OptixTraversableHandle gas_handle;
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // AABB build input
    CUdeviceptr d_aabb_buffer;
    const uint64_t d_aabb_buffer_size = aabbs.size() * sizeof(OptixAabb);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer),
                          d_aabb_buffer_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aabb_buffer), &aabbs[0],
                          d_aabb_buffer_size, cudaMemcpyHostToDevice));

    CUdeviceptr d_sbt_indices;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_indices),
                          nbPrimitives * sizeof(sbt_indices)));
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void*>(d_sbt_indices), sbt_indices.data(),
                   nbPrimitives * sizeof(sbt_indices), cudaMemcpyHostToDevice));

    OptixBuildInput aabb_input = {};

    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
    aabb_input.customPrimitiveArray.numPrimitives = nbPrimitives;
    aabb_input.customPrimitiveArray.flags = aabb_input_flags.data();
    aabb_input.customPrimitiveArray.numSbtRecords = nbPrimitives;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes =
        sizeof(uint32_t);
    aabb_input.customPrimitiveArray.primitiveIndexOffset = 0;
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options,
                                             &aabb_input, 1,
                                             &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas),
                          gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset =
        roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(
                              &d_buffer_temp_output_gas_and_compacted_size),
                          compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result =
        (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size +
                      compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(state.context,
                                0, // CUDA stream
                                &accel_options, &aabb_input,
                                1, // num build inputs
                                d_temp_buffer_gas,
                                gas_buffer_sizes.tempSizeInBytes,
                                d_buffer_temp_output_gas_and_compacted_size,
                                gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                                &emitProperty, // emitted property list
                                1              // num emitted properties
                                ));

    CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
    CUDA_CHECK(cudaFree((void*)d_aabb_buffer));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result,
                          sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer),
                       compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, gas_handle,
                                      state.d_gas_output_buffer,
                                      compacted_gas_size, &gas_handle));

        CUDA_CHECK(
            cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;

    PLUGIN_DEBUG("Registering OptiX STB Group Record");
    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record),
                          hitgroup_record_size * nbPrimitives));
    HitGroupRecord hg;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_prog_group, &hg));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), &hg,
                          hitgroup_record_size * nbPrimitives,
                          cudaMemcpyHostToDevice));

    state.sbt.hitgroupRecordBase = hitgroup_record;
    state.sbt.hitgroupRecordCount = nbPrimitives;
    state.sbt.hitgroupRecordStrideInBytes =
        nbPrimitives * sizeof(HitGroupRecord);
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
