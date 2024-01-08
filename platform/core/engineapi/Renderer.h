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

#include <platform/core/common/Api.h>
#include <platform/core/common/PropertyObject.h>
#include <platform/core/parameters/AnimationParameters.h>
#include <platform/core/parameters/RenderingParameters.h>

SERIALIZATION_ACCESS(Renderer)

namespace core
{
/**
 * @class Renderer
 * @extends PropertyObject
 * @brief Renderer class inherits from PropertyObject class
 * The Renderer class has methods to render a FrameBuffer, get variance, commit, and pick a point as well as virtual
 * methods to set camera and render frames in specific engines. It also contains a protected ScenePtr _scene variable.
 */
class Renderer : public PropertyObject
{
public:
    /**
     * @brief PickResult Struct containing hit boolean value and Vector3d pos
     * PickResult struct is used to retrieve information about whether the pick ray hit anything and if so the vector
     * position.
     * @struct PickResult
     */
    struct PickResult
    {
        bool hit{false}; // boolean value that defines if the ray hits anything
        Vector3d pos;    // vector position of the ray intersection
    };

    /**
     * @brief Virtual method to render a FrameBuffer
     * This method is implemented in specific engine renderers to draw the FrameBuffer.
     * @param frameBuffer Ptr to FrameBuffer that will be drawn
     */
    virtual void render(FrameBufferPtr frameBuffer) = 0;

    /**
     * @brief Get variance from previous render()
     * This method returns the variance from the previous render() call.
     * @return float Variance value
     */
    virtual float getVariance() const { return 0.f; }

    /**
     * @brief This virtual method is implemented in specific engine renderers to signal that rendering is complete.
     */
    virtual void commit() = 0;

    /**
     * @brief Set camera for renderer
     * This virtual method is implemented in specific engine renderers to set the camera for rendering.
     * @param camera CameraPtr of a camera object to set
     */
    virtual void setCamera(CameraPtr camera) = 0;

    /**
     * @brief Pick method
     * This method is used to pick a point on the scene and returns PickResult struct with hit boolean value and
     * position.
     * @param pickPos Vector2f with pick position coordinates
     * @return PickResult returns PickResult struct that contains boolean hit value and vector position
     */
    virtual PickResult pick(const Vector2f& /*pickPos*/) { return PickResult(); }

    /**
     * @brief Constructs the Renderer object with animationParameters and renderingParameters.
     * @param animationParameters const reference to AnimationParameters struct containing parameters for animation
     * @param renderingParameters const reference to RenderingParameters struct containing parameters for rendering
     */
    PLATFORM_API Renderer(const AnimationParameters& animationParameters,
                          const RenderingParameters& renderingParameters);
    /**
     * @brief Sets the _scene pointer to a specified ScenePtr.
     * @param scene ScenePtr of the scene object to set
     */
    void setEngine(Engine* engine) { _engine = engine; };

    /**
       Light source follow camera origin
    */
    PLATFORM_API bool getHeadLight() const { return _headLight; }
    PLATFORM_API void setHeadLight(const bool value) { _updateValue(_headLight, value); }

    /**
     * The maximum number of accumulation frames before engine signals to stop
     * continuation of rendering.
     *
     * @sa Engine::continueRendering()
     */
    PLATFORM_API void setMaxAccumFrames(const size_t value) { _updateValue(_maxAccumFrames, value); }
    PLATFORM_API size_t getMaxAccumFrames() const { return _maxAccumFrames; }

    /** Number of samples per pixel */
    PLATFORM_API uint32_t getSamplesPerPixel() const { return _spp; }
    PLATFORM_API void setSamplesPerPixel(const uint32_t value) { _updateValue(_spp, std::max(1u, value)); }

    /** Sub-sampling */
    PLATFORM_API uint32_t getSubsampling() const { return _subsampling; }
    PLATFORM_API void setSubsampling(const uint32_t subsampling)
    {
        _updateValue(_subsampling, std::max(1u, subsampling));
    }

    /** Background color */
    PLATFORM_API const Vector3d& getBackgroundColor() const { return _backgroundColor; }
    PLATFORM_API void setBackgroundColor(const Vector3d& value) { _updateValue(_backgroundColor, value); }

    /** If the rendering should be refined by accumulating multiple passes */
    PLATFORM_API bool getAccumulation() const { return _accumulation; }

protected:
    const AnimationParameters& _animationParameters;
    const RenderingParameters& _renderingParameters;
    Engine* _engine{nullptr};

    bool _accumulation{true};
    bool _headLight{true};
    Vector3d _backgroundColor{0., 0., 0.};
    uint32_t _spp{1};
    uint32_t _subsampling{1};
    size_t _maxAccumFrames{100};

private:
    SERIALIZATION_FRIEND(Renderer);
};
} // namespace core
