/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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
#include <platform/core/common/PropertyMap.h>
#include <platform/core/common/Statistics.h>

#include <functional>

namespace core
{
/**
 * @class Engine
 * @brief Provides an abstract implementation of a ray-tracing engine.
 *
 * The above code is a C++ class called "Engine", which provides an abstract implementation of a ray-tracing engine that
 * uses a 3rd party acceleration library. The engine holds a native implementation of a scene, a camera, a frame buffer
 * and one or several renderers according to the capabilities of the acceleration library.
 *
 * The class provides several API for engine-specific code, such as committing changes to the engine, executing engine
 * specific pre-render operations, executing engine specific post-render operations, getting the minimum frame size in
 * pixels supported by the engine, creating an engine-specific framebuffer, creating an engine-specific scene, creating
 * an engine-specific camera, and creating an engine-specific renderer.
 *
 * The constructor takes in a parameters manager that holds all engine parameters, such as geometry and rendering
 * parameters, and the class provides functions to retrieve the scene, frame buffer, camera, and renderer. The class
 * also provides a render function that renders the current scene and populates the frame buffer accordingly.
 *
 * In addition, there are several callback functions, such as triggerRender, which is called when a new frame shall be
 * triggered, and setKeepRunning and getKeepRunning functions, which allow the user to set and get a flag to continue or
 * stop rendering. The class also provides statistics and various functions to manage the frame buffers.
 *
 * Overall, the "Engine" class provides a flexible and extensible framework for implementing a ray-tracing engine using
 * a 3rd party acceleration library.
 */
class Engine
{
public:
    /**
     * @brief Commits changes to the engine. This includes scene modifications, camera modifications and renderer
     * modifications.
     */
    PLATFORM_API virtual void commit();

    /**
     * @brief Executes engine-specific pre-render operations.
     */
    PLATFORM_API virtual void preRender();

    /**
     * @brief Executes engine-specific post-render operations.
     */
    PLATFORM_API virtual void postRender();

    /**
     * @brief Returns the minimum frame size in pixels supported by this engine.
     *
     * @return Vector2ui The minimum frame size.
     */
    PLATFORM_API virtual Vector2ui getMinimumFrameSize() const = 0;

    /**
     * @brief Factory method to create an engine-specific framebuffer.
     *
     * @param name The name of the frame buffer.
     * @param frameSize The size of the frame buffer.
     * @param frameBufferFormat The frame buffer format.
     *
     * @return FrameBufferPtr The created frame buffer.
     */
    PLATFORM_API virtual FrameBufferPtr createFrameBuffer(const std::string& name, const Vector2ui& frameSize,
                                                          FrameBufferFormat frameBufferFormat) const = 0;

    /**
     * @brief Factory method to create an engine-specific scene.
     *
     * @param animationParameters The animation parameters.
     * @param geometryParameters The geometry parameters.
     * @param volumeParameters The volume parameters.
     *
     * @return ScenePtr The created scene.
     */
    PLATFORM_API virtual ScenePtr createScene(AnimationParameters& animationParameters,
                                              GeometryParameters& geometryParameters,
                                              VolumeParameters& volumeParameters) const = 0;

    /**
     * @brief Factory method to create an engine-specific camera.
     *
     * @return CameraPtr The created camera.
     */
    PLATFORM_API virtual CameraPtr createCamera() const = 0;

    /**
     * @brief Factory method to create an engine-specific renderer.
     *
     * @param animationParameters The animation parameters.
     * @param renderingParameters The rendering parameters.
     *
     * @return RendererPtr The created renderer.
     */
    PLATFORM_API virtual RendererPtr createRenderer(const AnimationParameters& animationParameters,
                                                    const RenderingParameters& renderingParameters) const = 0;

    /**
     * @brief Engine Constructor.
     *
     * @param parametersManager The parameter manager that holds all engine parameters.
     */
    PLATFORM_API explicit Engine(ParametersManager& parametersManager);

    PLATFORM_API virtual ~Engine() = default;

    /**
     * @brief Renders the current scene and populates the frame buffer accordingly.
     */
    PLATFORM_API void render();

    /**
     * @brief Returns the scene.
     *
     * @return Scene& The current scene.
     */
    PLATFORM_API Scene& getScene() { return *_scene; }

    /**
     * @brief Returns the frame buffer.
     *
     * @return FrameBuffer& The frame buffer.
     */
    PLATFORM_API FrameBuffer& getFrameBuffer() { return *_frameBuffers[0]; }

    /**
     * @brief Returns the camera.
     *
     * @return Camera& The camera.
     */
    PLATFORM_API const Camera& getCamera() const { return *_camera; }
    PLATFORM_API Camera& getCamera() { return *_camera; }

    /**
     * @brief Returns the renderer.
     *
     * @return Renderer& The renderer.
     */
    PLATFORM_API Renderer& getRenderer();

    /**
     * @brief Callback when a new frame shall be triggered. Currently called by event plugins Deflect and Rockets.
     */
    PLATFORM_API std::function<void()> triggerRender{[] {}};

    /**
     * @brief Sets a flag to continue or stop rendering.
     *
     * @param keepRunning The flag to set.
     */
    PLATFORM_API void setKeepRunning(bool keepRunning) { _keepRunning = keepRunning; }

    /**
     * @brief Returns a boolean indicating whether the user wants to continue rendering.
     *
     * @return bool Value indicating whether the user wants to continue rendering. True if they do, false otherwise.
     */
    PLATFORM_API bool getKeepRunning() const { return _keepRunning; }

    /**
     * @brief Returns statistics information.
     *
     * @return Statistics& Statistics information.
     */
    PLATFORM_API Statistics& getStatistics() { return _statistics; }

    /**
     * @brief Returns a boolean indicating whether render calls shall be continued based on current accumulation
     * settings.
     *
     * @return bool Value indicating whether render calls shall be continued based on current accumulation settings.
     */
    PLATFORM_API bool continueRendering() const;

    /**
     * @brief Returns the parameter manager.
     *
     * @return const auto& The parameter manager.
     */
    PLATFORM_API const auto& getParametersManager() const { return _parametersManager; }

    /**
     * @brief Adds a frame buffer to the list to be filled during rendering.
     *
     * @param frameBuffer The frame buffer to add.
     */
    PLATFORM_API void addFrameBuffer(FrameBufferPtr frameBuffer);

    /**
     * @brief Removes a frame buffer from the list of buffers that are filled during rendering.
     *
     * @param frameBuffer The frame buffer to remove.
     */
    PLATFORM_API void removeFrameBuffer(FrameBufferPtr frameBuffer);

    /**
     * @brief Returns all registered frame buffers that are used during rendering.
     *
     * @return const std::vector<FrameBufferPtr>& A vector containing all registered frame buffers.
     */
    PLATFORM_API const std::vector<FrameBufferPtr>& getFrameBuffers() const { return _frameBuffers; }

    /**
     * @brief Clears all frame buffers.
     */
    PLATFORM_API void clearFrameBuffers();

    /**
     * @brief Resets all frame buffers.
     */
    PLATFORM_API void resetFrameBuffers();

    /**
     * @brief Adds a new renderer type with optional properties.
     *
     * @param name The renderer type name.
     * @param properties The properties.
     */
    PLATFORM_API void addRendererType(const std::string& name, const PropertyMap& properties = {});

    /**
     * @brief Returns all renderer types.
     *
     * @return const strings& A vector containing all renderer types.
     */
    PLATFORM_API const strings& getRendererTypes() const { return _rendererTypes; };

    /**
     * @brief Adds a new camera type with optional properties.
     *
     * @param name The camera type name.
     * @param properties The properties.
     */
    PLATFORM_API void addCameraType(const std::string& name, const PropertyMap& properties = {});

protected:
    ParametersManager& _parametersManager;
    ScenePtr _scene;
    CameraPtr _camera;
    RendererPtr _renderer;
    std::vector<FrameBufferPtr> _frameBuffers;
    Statistics _statistics;
    strings _rendererTypes;
    bool _keepRunning{true};
};
} // namespace core
