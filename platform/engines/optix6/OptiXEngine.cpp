/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
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

#include <platform/engines/optix6/OptiX6Engine_generated_Advanced.cu.ptx.h>
#include <platform/engines/optix6/OptiX6Engine_generated_Basic.cu.ptx.h>

#include <platform/core/common/input/KeyboardHandler.h>
#include <platform/core/parameters/ParametersManager.h>

#include "Logs.h"
#include "OptiXAnaglyphCamera.h"
#include "OptiXCamera.h"
#include "OptiXEngine.h"
#include "OptiXFrameBuffer.h"
#include "OptiXOrthographicCamera.h"
#include "OptiXPerspectiveCamera.h"
#include "OptiXProperties.h"
#include "OptiXRenderer.h"
#include "OptiXScene.h"

namespace core
{
namespace engine
{
namespace optix
{
OptiXEngine::OptiXEngine(ParametersManager& parametersManager)
    : Engine(parametersManager)
{
    PLUGIN_INFO("Initializing OptiX");
    _initializeContext();

    PLUGIN_INFO("Initializing scene");
    _scene = std::make_shared<OptiXScene>(_parametersManager.getAnimationParameters(),
                                          _parametersManager.getGeometryParameters(),
                                          _parametersManager.getVolumeParameters());

    PLUGIN_INFO("Initializing renderers");
    _createRenderers();

    PLUGIN_INFO("Initializing cameras");
    _createCameras();

    PLUGIN_INFO("Engine initialization complete");
}

OptiXEngine::~OptiXEngine()
{
    _scene.reset();
    for (auto& fb : _frameBuffers)
        fb.reset();
    _renderer.reset();
    _camera.reset();

    _frameBuffers.clear();
}

void OptiXEngine::_initializeContext()
{
    // Set up context
    auto context = OptiXContext::get().getOptixContext();
    if (!context)
        PLUGIN_THROW(std::runtime_error("Failed to initialize OptiX"));
}

void OptiXEngine::_createCameras()
{
    _camera = createCamera();

    const bool isStereo = _parametersManager.getApplicationParameters().isStereo();
    Property stereoProperty{CAMERA_PROPERTY_STEREO.name, isStereo, CAMERA_PROPERTY_STEREO.metaData};
    Property aspect = CAMERA_PROPERTY_ASPECT_RATIO;
    aspect.markReadOnly();

    OptiXContext& context = OptiXContext::get();

    {
        PLUGIN_INFO("Registering '" << CAMERA_PROPERTY_TYPE_PERSPECTIVE << "' camera");
        PropertyMap properties;
        properties.setProperty(COMMON_PROPERTY_EXPOSURE);
        properties.setProperty(CAMERA_PROPERTY_FIELD_OF_VIEW);
        properties.setProperty(aspect);
        properties.setProperty(CAMERA_PROPERTY_APERTURE_RADIUS);
        properties.setProperty(CAMERA_PROPERTY_FOCAL_DISTANCE);
        properties.setProperty(CAMERA_PROPERTY_NEAR_CLIP);
        properties.setProperty(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES);
        properties.setProperty(stereoProperty);
        properties.setProperty(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE);

        auto camera = std::make_shared<OptiXPerspectiveCamera>();
        context.addCamera(CAMERA_PROPERTY_TYPE_PERSPECTIVE, camera);
        addCameraType(CAMERA_PROPERTY_TYPE_PERSPECTIVE, properties);
    }

    {
        PLUGIN_INFO("Registering '" << CAMERA_PROPERTY_TYPE_ORTHOGRAPHIC << "' camera");
        PropertyMap properties;
        properties.setProperty(COMMON_PROPERTY_EXPOSURE);
        properties.setProperty(CAMERA_PROPERTY_HEIGHT);
        properties.setProperty(aspect);
        properties.setProperty(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES);

        auto camera = std::make_shared<OptiXOrthographicCamera>();
        context.addCamera(CAMERA_PROPERTY_TYPE_ORTHOGRAPHIC, camera);
        addCameraType(CAMERA_PROPERTY_TYPE_ORTHOGRAPHIC, properties);
    }

    {
        PLUGIN_INFO("Registering '" << CAMERA_PROPERTY_TYPE_ANAGLYPH << "' camera");
        PropertyMap properties;
        properties.setProperty(COMMON_PROPERTY_EXPOSURE);
        properties.setProperty(CAMERA_PROPERTY_FIELD_OF_VIEW);
        properties.setProperty(aspect);
        properties.setProperty(CAMERA_PROPERTY_APERTURE_RADIUS);
        properties.setProperty(CAMERA_PROPERTY_FOCAL_DISTANCE);
        properties.setProperty(CAMERA_PROPERTY_NEAR_CLIP);
        properties.setProperty(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES);
        properties.setProperty(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE);

        auto camera = std::make_shared<OptiXAnaglyphCamera>();
        context.addCamera(CAMERA_PROPERTY_TYPE_ANAGLYPH, camera);
        addCameraType(CAMERA_PROPERTY_TYPE_ANAGLYPH, properties);
    }
}

void OptiXEngine::_createRenderers()
{
    _renderer = std::make_shared<OptiXRenderer>(_parametersManager.getAnimationParameters(),
                                                _parametersManager.getRenderingParameters());
    _renderer->setScene(_scene);
    OptiXContext& context = OptiXContext::get();

    { // Advanced renderer
        PLUGIN_INFO("Registering '" << RENDERER_PROPERTY_TYPE_ADVANCED << "' renderer");
        const std::string CUDA_ADVANCED_SIMULATION_RENDERER = OptiX6Engine_generated_Advanced_cu_ptx;

        auto renderer = std::make_shared<OptixShaderProgram>();
        renderer->closest_hit =
            context.getOptixContext()->createProgramFromPTXString(CUDA_ADVANCED_SIMULATION_RENDERER,
                                                                  OPTIX_CUDA_FUNCTION_CLOSEST_HIT_RADIANCE);
        renderer->closest_hit_textured =
            context.getOptixContext()->createProgramFromPTXString(CUDA_ADVANCED_SIMULATION_RENDERER,
                                                                  OPTIX_CUDA_FUNCTION_CLOSEST_HIT_RADIANCE_TEXTURED);
        renderer->any_hit = context.getOptixContext()->createProgramFromPTXString(CUDA_ADVANCED_SIMULATION_RENDERER,
                                                                                  OPTIX_CUDA_FUNCTION_ANY_HIT_SHADOW);
        renderer->exception_program =
            context.getOptixContext()->createProgramFromPTXString(CUDA_ADVANCED_SIMULATION_RENDERER,
                                                                  OPTIX_CUDA_FUNCTION_EXCEPTION);
        context.getOptixContext()->setExceptionProgram(0, renderer->exception_program);
        context.addRenderer(RENDERER_PROPERTY_TYPE_ADVANCED, renderer);

        PropertyMap properties;
        properties.setProperty(RENDERER_PROPERTY_ALPHA_CORRECTION);
        properties.setProperty(RENDERER_PROPERTY_MAX_DISTANCE_TO_SECONDARY_MODEL);
        properties.setProperty(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_RAY_LENGTH);
        properties.setProperty(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_STRENGTH);
        properties.setProperty(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_SAMPLES);
        properties.setProperty(RENDERER_PROPERTY_SHADOW_INTENSITY);
        properties.setProperty(RENDERER_PROPERTY_SOFT_SHADOW_STRENGTH);
        properties.setProperty(RENDERER_PROPERTY_SHADOW_SAMPLES);
        properties.setProperty(COMMON_PROPERTY_EXPOSURE);
        properties.setProperty(RENDERER_PROPERTY_EPSILON_MULTIPLIER);
        properties.setProperty(RENDERER_PROPERTY_FOG_START);
        properties.setProperty(RENDERER_PROPERTY_FOG_THICKNESS);
        properties.setProperty(RENDERER_PROPERTY_MAX_RAY_DEPTH);
        properties.setProperty(RENDERER_PROPERTY_SHOW_BACKGROUND);
        properties.setProperty(RENDERER_PROPERTY_MATRIX_FILTER);
        addRendererType(RENDERER_PROPERTY_TYPE_ADVANCED, properties);
    }

    { // Basic simulation / Basic renderer
        PLUGIN_INFO("Registering '" << RENDERER_PROPERTY_TYPE_BASIC << "' renderer");
        const std::string CUDA_BASIC_SIMULATION_RENDERER = OptiX6Engine_generated_Basic_cu_ptx;

        auto renderer = std::make_shared<OptixShaderProgram>();
        renderer->closest_hit =
            context.getOptixContext()->createProgramFromPTXString(CUDA_BASIC_SIMULATION_RENDERER,
                                                                  OPTIX_CUDA_FUNCTION_CLOSEST_HIT_RADIANCE);
        renderer->closest_hit_textured =
            context.getOptixContext()->createProgramFromPTXString(CUDA_BASIC_SIMULATION_RENDERER,
                                                                  OPTIX_CUDA_FUNCTION_CLOSEST_HIT_RADIANCE_TEXTURED);
        renderer->any_hit = context.getOptixContext()->createProgramFromPTXString(CUDA_BASIC_SIMULATION_RENDERER,
                                                                                  OPTIX_CUDA_FUNCTION_ANY_HIT_SHADOW);
        renderer->exception_program =
            context.getOptixContext()->createProgramFromPTXString(CUDA_BASIC_SIMULATION_RENDERER,
                                                                  OPTIX_CUDA_FUNCTION_EXCEPTION);

        context.addRenderer(RENDERER_PROPERTY_TYPE_BASIC, renderer);

        PropertyMap properties;
        properties.setProperty(COMMON_PROPERTY_EXPOSURE);
        properties.setProperty(RENDERER_PROPERTY_SHOW_BACKGROUND);
        addRendererType(RENDERER_PROPERTY_TYPE_BASIC, properties);
    }
}

ScenePtr OptiXEngine::createScene(AnimationParameters& animationParameters, GeometryParameters& geometryParameters,
                                  VolumeParameters& volumeParameters) const
{
    return std::make_shared<OptiXScene>(animationParameters, geometryParameters, volumeParameters);
}

FrameBufferPtr OptiXEngine::createFrameBuffer(const std::string& name, const Vector2ui& frameSize,
                                              FrameBufferFormat frameBufferFormat) const
{
    return std::make_shared<OptiXFrameBuffer>(name, frameSize, frameBufferFormat,
                                              _parametersManager.getRenderingParameters());
}

RendererPtr OptiXEngine::createRenderer(const AnimationParameters& animationParameters,
                                        const RenderingParameters& renderingParameters) const
{
    return std::make_shared<OptiXRenderer>(animationParameters, renderingParameters);
}

CameraPtr OptiXEngine::createCamera() const
{
    return std::make_shared<OptiXCamera>();
}

void OptiXEngine::commit()
{
    Engine::commit();
}

Vector2ui OptiXEngine::getMinimumFrameSize() const
{
    return {1, 1};
}
} // namespace optix
} // namespace engine
} // namespace core

extern "C" core::Engine* core_engine_create(int, const char**, core::ParametersManager& parametersManager)
{
    PLUGIN_INFO("");
    PLUGIN_INFO("   _|_|                _|      _|  _|      _|        _|_|_|  ");
    PLUGIN_INFO(" _|    _|  _|_|_|    _|_|_|_|        _|  _|        _|        ");
    PLUGIN_INFO(" _|    _|  _|    _|    _|      _|      _|          _|_|_|    ");
    PLUGIN_INFO(" _|    _|  _|    _|    _|      _|    _|  _|        _|    _|  ");
    PLUGIN_INFO("   _|_|    _|_|_|        _|_|  _|  _|      _|        _|_|    ");
    PLUGIN_INFO("           _|                                                ");
    PLUGIN_INFO("           _|                                                ");
    PLUGIN_INFO("");

    return new core::engine::optix::OptiXEngine(parametersManager);
}