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

#include <ospray.h>
#include <platform/core/engineapi/FrameBuffer.h>

#include <mutex>

namespace core
{
namespace engine
{
namespace ospray
{
class OSPRayFrameBuffer : public FrameBuffer
{
public:
    OSPRayFrameBuffer(const std::string& name, const Vector2ui& frameSize, const FrameBufferFormat frameBufferFormat);
    ~OSPRayFrameBuffer();

    void clear() final;
    void resize(const Vector2ui& frameSize) final;
    void map() final;
    void unmap() final;
    void setAccumulation(const bool accumulation) final;
    void setFormat(FrameBufferFormat frameBufferFormat) final;
    void setSubsampling(const size_t) final;
    Vector2ui getSize() const final { return _useSubsampling() ? _subsamplingSize() : _frameSize; }
    std::unique_lock<std::mutex> getScopeLock() { return std::unique_lock<std::mutex>(_mapMutex); }
    const uint8_t* getColorBuffer() const final { return _colorBuffer; }
    const float* getFloatBuffer() const final { return _floatBuffer; }
    OSPFrameBuffer impl() { return _currentFB(); }
    void createPixelOp(const std::string& name) final;
    void updatePixelOp(const PropertyMap& properties) final;

private:
    void _recreate();
    void _recreateSubsamplingBuffer();
    void _unmapUnsafe();
    void _mapUnsafe();
    bool _useSubsampling() const;
    OSPFrameBuffer _currentFB() const;
    Vector2ui _subsamplingSize() const;

    OSPFrameBuffer _frameBuffer{nullptr};
    OSPFrameBuffer _subsamplingFrameBuffer{nullptr};
    uint8_t* _colorBuffer{nullptr};
    float* _floatBuffer{nullptr};
    OSPPixelOp _pixelOp{nullptr};
    size_t _subsamplingFactor{1};

    // protect map/unmap vs ospRenderFrame
    std::mutex _mapMutex;
};
} // namespace ospray
} // namespace engine
} // namespace core