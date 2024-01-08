/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
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

#pragma once

#include <optixu/optixpp_namespace.h>
#include <platform/core/engineapi/FrameBuffer.h>

#include <mutex>

namespace core
{
namespace engine
{
namespace optix
{
/**
   OptiX specific frame buffer

   This object is the OptiX specific implementation of a frame buffer
*/
class OptiXFrameBuffer : public FrameBuffer
{
public:
    OptiXFrameBuffer(const std::string& name, const Vector2ui& size, FrameBufferFormat frameBufferFormat,
                     const RenderingParameters& renderingParameters);
    ~OptiXFrameBuffer();

    void resize(const Vector2ui& size) final;
    void map() final;
    void unmap() final;
    void setAccumulation(const bool accumulation) final;

    std::unique_lock<std::mutex> getScopeLock() { return std::unique_lock<std::mutex>(_mapMutex); }
    const uint8_t* getColorBuffer() const final { return _colorBuffer; }
    const float* getFloatBuffer() const final { return _floatBuffer; }

private:
    void _cleanup();
    void _mapUnsafe();
    void _unmapUnsafe();

    ::optix::Buffer _outputBuffer{nullptr};
    ::optix::Buffer _accumBuffer{nullptr};
    ::optix::Buffer _tonemappedBuffer{nullptr};
    ::optix::Buffer _denoisedBuffer{nullptr};

    uint8_t* _colorBuffer{nullptr};
    float* _floatBuffer{nullptr};
    void* _imageData{nullptr};
    void* _colorData{nullptr};
    void* _floatData{nullptr};

    // Post processing
    void _initializePostProcessingStages();
    ::optix::CommandList _commandListWithDenoiser{nullptr};
    ::optix::CommandList _commandListWithDenoiserAndToneMapper{nullptr};

    ::optix::PostprocessingStage _tonemapStage{nullptr};
    ::optix::PostprocessingStage _denoiserStage{nullptr};
    ::optix::PostprocessingStage _denoiserWithMappingStage{nullptr};

    uint64_t _accumulationFrameNumber{1u};

    bool _postprocessingStagesInitialized{false};

    const RenderingParameters& _renderingParameters;

    // protect map/unmap
    std::mutex _mapMutex;
};
} // namespace optix
} // namespace engine
} // namespace core