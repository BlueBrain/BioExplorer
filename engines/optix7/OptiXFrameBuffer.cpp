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

#include "OptiXFrameBuffer.h"
#include "OptiXContext.h"

#include <brayns/common/log.h>

#include <optix.h>

namespace brayns
{
OptiXFrameBuffer::OptiXFrameBuffer(const std::string& name, const Vector2ui& frameSize,
                                   FrameBufferFormat frameBufferFormat)
    : FrameBuffer(name, frameSize, frameBufferFormat)
{
    resize(frameSize);
}

OptiXFrameBuffer::~OptiXFrameBuffer()
{
    auto lock = getScopeLock();
    _unmapUnsafe();
}

void OptiXFrameBuffer::resize(const Vector2ui& frameSize)
{
    if (glm::compMul(frameSize) == 0)
        throw std::runtime_error("Invalid size for framebuffer resize");

    if (_frameBuffer && getSize() == frameSize)
        return;

    _frameSize = frameSize;

    _recreate();
}

void OptiXFrameBuffer::_recreate()
{
    BRAYNS_DEBUG("Creating frame buffer...");
    auto lock = getScopeLock();
    auto& state = OptiXContext::getInstance().getState();

    state.params.width = _frameSize.x;
    state.params.height = _frameSize.y;

    if (_frameBuffer)
    {
        _unmapUnsafe();
        _frameBuffer->resize(state.params.width, state.params.height);
    }
    else
    {
        _frameBuffer = new sutil::CUDAOutputBuffer<uchar4>(sutil::CUDAOutputBufferType::CUDA_DEVICE, state.params.width,
                                                           state.params.height);
        _frameBuffer->setStream(state.stream);
    }

    CUDA_CHECK(cudaFree(state.params.accum_buffer));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.params.accum_buffer),
                          state.params.width * state.params.height * sizeof(float4)));
    state.params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    clear();
    BRAYNS_DEBUG("Frame buffer created");
}

void OptiXFrameBuffer::map()
{
    _mapMutex.lock();
    _mapUnsafe();
}

void OptiXFrameBuffer::_mapUnsafe()
{
    // Launch
    auto& state = OptiXContext::getInstance().getState();
    state.params.frame_buffer = _frameBuffer->map();
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(state.d_params), &state.params, sizeof(Params),
                               cudaMemcpyHostToDevice, state.stream));

    OPTIX_CHECK(optixLaunch(state.pipeline, state.stream, reinterpret_cast<CUdeviceptr>(state.d_params), sizeof(Params),
                            &state.sbt, state.params.width, state.params.height, 1));
    _frameBuffer->unmap();
    CUDA_SYNC_CHECK();

    sutil::ImageBuffer buffer;
    buffer.data = _frameBuffer->getHostPointer();
    buffer.width = state.params.width;
    buffer.height = state.params.height;

    switch (_frameBufferFormat)
    {
    case FrameBufferFormat::rgba_i8:
    case FrameBufferFormat::bgra_i8:
        buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
        break;
    case FrameBufferFormat::rgb_f32:
        buffer.pixel_format = sutil::BufferImageFormat::FLOAT4;
        break;
    default:
        BRAYNS_THROW(std::runtime_error("Unsupported frame buffer format"));
    }

    _imageData = buffer.data;

    switch (_frameBufferFormat)
    {
    case FrameBufferFormat::rgba_i8:
    case FrameBufferFormat::bgra_i8:
        _colorBuffer = (uint8_t*)(_imageData);
        break;
    case FrameBufferFormat::rgb_f32:
        _depthBuffer = (float*)_imageData;
        break;
    default:
        BRAYNS_THROW(std::runtime_error("Unsupported frame buffer format"));
    }
}

void OptiXFrameBuffer::unmap()
{
    _unmapUnsafe();
    _mapMutex.unlock();
}

void OptiXFrameBuffer::_unmapUnsafe()
{
    if (_frameBufferFormat == FrameBufferFormat::none)
        return;

    _frameBuffer->unmap();
    _colorBuffer = nullptr;
    _depthBuffer = nullptr;
}

void OptiXFrameBuffer::setAccumulation(const bool accumulation)
{
    if (_accumulation != accumulation)
    {
        FrameBuffer::setAccumulation(accumulation);
        _recreate();
    }
}

} // namespace brayns
