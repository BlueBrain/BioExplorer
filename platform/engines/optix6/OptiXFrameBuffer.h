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
    const uint8_t* getColorBuffer() const final { return _colorDataBuffer; }
    const float* getFloatBuffer() const final { return _floatDataBuffer; }
    const float* getDepthBuffer() const { return _depthDataBuffer; }

private:
    void _cleanup();
    void _mapUnsafe();
    void _unmapUnsafe();

    ::optix::Buffer _outputBuffer{nullptr};
    ::optix::Buffer _accumBuffer{nullptr};
    ::optix::Buffer _depthBuffer{nullptr};
    ::optix::Buffer _tonemappedBuffer{nullptr};
    ::optix::Buffer _denoisedBuffer{nullptr};

    uint8_t* _colorDataBuffer{nullptr};
    float* _floatDataBuffer{nullptr};
    float* _depthDataBuffer{nullptr};
    void* _imageData{nullptr};
    void* _colorData{nullptr};
    void* _floatData{nullptr};
    void* _depthData{nullptr};

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