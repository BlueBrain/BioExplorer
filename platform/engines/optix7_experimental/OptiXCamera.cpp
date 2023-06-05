/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Core <https://github.com/BlueBrain/Core>
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

#include "OptiXCamera.h"
#include "Logs.h"

#include <optix_function_table_definition.h>

#include <Exception.h>

namespace
{
const std::string CUDA_CLIP_PLANES = "clip_planes";
const std::string CUDA_NB_CLIP_PLANES = "nb_clip_planes";
} // namespace

namespace core
{
OptiXCamera::OptiXCamera()
{
    auto& state = OptiXContext::getInstance().getState();

    // ---------------------------------------------------------------------------------------------
    PLUGIN_DEBUG("Registering OptiX SBT Ray Generation Program Record");
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
    RayGenRecord rg_sbt;
    rg_sbt.data = {};
    const Vector3f eye = getPosition();
    rg_sbt.data.cam_eye = {eye.x, eye.y, eye.z};
    rg_sbt.data.camera_u = {_u.x, _u.y, _u.z};
    rg_sbt.data.camera_v = {_v.x, _v.y, _v.z};
    rg_sbt.data.camera_w = {_w.x, _w.y, _w.z};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

    state.sbt.raygenRecord = raygen_record;

    // ---------------------------------------------------------------------------------------------
    // Miss program record
    // ---------------------------------------------------------------------------------------------
    PLUGIN_DEBUG("Registering OptiX SBT Miss Program Record");
    size_t sizeof_miss_record = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&_d_miss_record), sizeof_miss_record * RAY_TYPE_COUNT));

    MissRecord ms_sbt[RAY_TYPE_COUNT];
    optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt[0]);
    const auto bgColor = state.params.ambient_light_color;
    for (uint32_t i = 0; i < RAY_TYPE_COUNT; ++i)
        ms_sbt[i].data = {bgColor.x, bgColor.y, bgColor.z}; // Background color

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(_d_miss_record), ms_sbt, sizeof_miss_record * RAY_TYPE_COUNT,
                          cudaMemcpyHostToDevice));

    state.sbt.missRecordBase = _d_miss_record;
    state.sbt.missRecordCount = RAY_TYPE_COUNT;
    state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof_miss_record);
}

OptiXCamera::~OptiXCamera() {}

void OptiXCamera::commit()
{
    if (_currentCamera != getCurrentType())
    {
        _currentCamera = getCurrentType();
        OptiXContext::getInstance().setCamera(_currentCamera);
    }

    const auto position = getPosition();
    const auto up = glm::rotate(getOrientation(), Vector3d(0, 1, 0));

    float ulen, vlen, wlen;
    _w = getTarget() - position;

    wlen = glm::length(_w);
    _u = normalize(glm::cross(_w, Vector3f(up)));
    _v = normalize(glm::cross(_u, _w));

    vlen = wlen * tanf(0.5f * getPropertyOrValue<double>("fovy", 45.0) * M_PI / 180.f);
    _v *= vlen;
    ulen = vlen * getPropertyOrValue<double>("aspect", 1.0);
    _u *= ulen;

    _commitToOptiX();

#if 0
    auto cameraProgram = OptiXContext::getInstance().getCamera(_currentCamera);

    auto context = OptiXContext::getInstance().getOptixContext();

    cameraProgram->commit(*this, context);

    if (_clipPlanesBuffer)
        _clipPlanesBuffer->destroy();

    const size_t numClipPlanes = _clipPlanes.size();
    if (numClipPlanes > 0)
    {
        Vector4fs buffer;
        buffer.reserve(numClipPlanes);
        for (const auto& clipPlane : _clipPlanes)
            buffer.push_back({static_cast<float>(clipPlane[0]),
                              static_cast<float>(clipPlane[1]),
                              static_cast<float>(clipPlane[2]),
                              static_cast<float>(clipPlane[3])});

        _clipPlanesBuffer =
            context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4,
                                  numClipPlanes);
        memcpy(_clipPlanesBuffer->map(), buffer.data(),
               numClipPlanes * sizeof(Vector4f));
        _clipPlanesBuffer->unmap();
    }
    else
    {
        // Create empty buffer to avoid unset variable exception in cuda
        _clipPlanesBuffer =
            context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, 1);
    }

    context[CUDA_CLIP_PLANES]->setBuffer(_clipPlanesBuffer);
    context[CUDA_NB_CLIP_PLANES]->setUint(numClipPlanes);
#endif
}

void OptiXCamera::_commitToOptiX()
{
    auto& state = OptiXContext::getInstance().getState();

    RayGenRecord rg;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg));

    const Vector3f eye = getPosition();
    rg.data.cam_eye = {eye.x, eye.y, eye.z};
    rg.data.camera_u = {_u.x, _u.y, _u.z};
    rg.data.camera_v = {_v.x, _v.y, _v.z};
    rg.data.camera_w = {_w.x, _w.y, _w.z};
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void*>(state.sbt.raygenRecord), &rg, sizeof(RayGenRecord), cudaMemcpyHostToDevice));

    // Update miss record
    MissRecord ms_sbt[RAY_TYPE_COUNT];
    size_t sizeof_miss_record = sizeof(MissRecord);
    optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt[0]);
    const auto bgColor = state.params.ambient_light_color;
    for (uint32_t i = 0; i < RAY_TYPE_COUNT; ++i)
        ms_sbt[i].data.bg_color = {bgColor.x, bgColor.y, bgColor.z};
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(_d_miss_record), ms_sbt, sizeof_miss_record * RAY_TYPE_COUNT,
                          cudaMemcpyHostToDevice));
}

} // namespace core
