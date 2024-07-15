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

#include <platform/core/common/Api.h>
#include <platform/core/common/BaseObject.h>
#include <platform/core/common/Types.h>
#include <platform/core/common/utils/ImageUtils.h>

#include <OpenImageIO/imagebuf.h>

OIIO_NAMESPACE_USING

namespace core
{
/**
 * @class FrameBuffer
 * @extends BaseObject
 * @brief This class represents a frame buffer for an engine specific code. It provides an API
 * for utilizing and manipulating the frame buffer in various ways.
 */
class FrameBuffer : public BaseObject
{
public:
    /**
     * @brief FrameBuffer constructor
     *
     * Construct a new FrameBuffer object
     *
     * @param name The name of the frame buffer.
     * @param frameSize The initial size of the frame buffer.
     * @param frameBufferFormat The format of the frame buffer.
     */
    PLATFORM_API FrameBuffer(const std::string& name, const Vector2ui& frameSize, FrameBufferFormat frameBufferFormat);

    /**
     *@brief Map the buffer for reading with get*Buffer().
     *
     */
    PLATFORM_API virtual void map() = 0;

    /**
     *
     * @brief Unmap the buffer for reading with get*Buffer().
     *
     */
    PLATFORM_API virtual void unmap() = 0;

    /**
     * @brief Get the Color Buffer object
     *
     * @return const uint8_t* The color buffer.
     */
    PLATFORM_API virtual const uint8_t* getColorBuffer() const = 0;

    /**
     * @brief Get the Float Buffer object
     *
     * @return const float* The depth buffer.
     */
    PLATFORM_API virtual const float* getFloatBuffer() const = 0;

    /**
     * @brief Resize the framebuffer to the new size.
     *
     * @param frameSize The frame buffer size to be set.
     */
    PLATFORM_API virtual void resize(const Vector2ui& frameSize) = 0;

    /**
     * @brief Clear the framebuffer.
     *
     */
    PLATFORM_API virtual void clear() { _accumFrames = 0; }

    /**
     * @brief Get the Size object
     *
     * @return Vector2ui The current framebuffer size.
     */
    PLATFORM_API virtual Vector2ui getSize() const { return _frameSize; }

    /**
     * @brief Enable/disable accumulation state on the framebuffer.
     *
     * @param accumulation The accumulation state to be set.
     */
    PLATFORM_API virtual void setAccumulation(const bool accumulation) { _accumulation = accumulation; }

    /**
     * @brief Set a new framebuffer format.
     *
     * @param frameBufferFormat The new frame buffer format to be set.
     */
    PLATFORM_API virtual void setFormat(FrameBufferFormat frameBufferFormat) { _frameBufferFormat = frameBufferFormat; }

    /**
     * @brief Set a new subsampling with a factor from 1 to x of the current size.
     *
     * @param size_t The size to be set.
     */
    PLATFORM_API virtual void setSubsampling(const size_t) {}

    /**
     * @brief Create and set a pixelop (pre/post filter) on the framebuffer.
     *
     * @param name The name of the pixelOp.
     */
    PLATFORM_API virtual void createPixelOp(const std::string& /*name*/){};

    /**
     * @brief Update the current pixelop with the given properties.
     *
     * @param properties The properties to be updated.
     */
    PLATFORM_API virtual void updatePixelOp(const PropertyMap& /*properties*/){};

    /**
     * @brief Get the Color Depth object
     *
     * @return size_t The color depth.
     */
    PLATFORM_API size_t getColorDepth() const;

    /**
     * @brief Get the Frame Size object
     *
     * @return const Vector2ui& The current frame size.
     */
    PLATFORM_API const Vector2ui& getFrameSize() const { return _frameSize; }

    /**
     * @brief Get the Accumulation object
     *
     * @return bool The current accumulation state.
     */
    PLATFORM_API bool getAccumulation() const { return _accumulation; }

    /**
     * @brief Get the Frame Buffer Format object
     *
     * @return FrameBufferFormat The current frame buffer format.
     */
    PLATFORM_API FrameBufferFormat getFrameBufferFormat() const { return _frameBufferFormat; }

    /**
     * @brief Get the Name object
     *
     * @return const std::string& The name of the frame buffer.
     */
    PLATFORM_API const std::string& getName() const { return _name; }

    /**
     * @brief Increment the accumulation frames.
     *
     */
    PLATFORM_API void incrementAccumFrames() { ++_accumFrames; }

    /**
     * @brief Get the number of accumulation frames.
     *
     * @return size_t The number of accumulated frames.
     */
    PLATFORM_API size_t numAccumFrames() const { return _accumFrames; }

    /**
     * @brief Get the Image object
     *
     * @return ImageBuf The OpenImageIO object.
     */
    PLATFORM_API ImageBuf getImage();

    /**
     * @brief Set the Accumulation Type object
     *
     * @param accumulationType The accumulation type to be set.
     */
    PLATFORM_API void setAccumulationType(const AccumulationType accumulationType)
    {
        _accumulationType = accumulationType;
    }

    /**
     * @brief Get the Accumulation Type object
     *
     * @return AccumulationType The current accumulation type.
     */
    PLATFORM_API AccumulationType getAccumulationType() const { return _accumulationType; }

protected:
    const std::string _name;
    Vector2ui _frameSize;
    FrameBufferFormat _frameBufferFormat;
    bool _accumulation{true};
    AccumulationType _accumulationType{AccumulationType::linear};
    std::atomic_size_t _accumFrames{0};
};
} // namespace core
