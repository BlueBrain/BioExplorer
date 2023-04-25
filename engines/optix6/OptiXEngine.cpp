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

#include <engines/optix6/braynsOptix6Engine_generated_Basic.cu.ptx.h>
#include <engines/optix6/braynsOptix6Engine_generated_BioExplorer.cu.ptx.h>

#include <brayns/common/input/KeyboardHandler.h>
#include <brayns/parameters/ParametersManager.h>

#include "Logs.h"
#include "OptiXCamera.h"
#include "OptiXEngine.h"
#include "OptiXFrameBuffer.h"
#include "OptiXOpenDeckCamera.h"
#include "OptiXOrthographicCamera.h"
#include "OptiXPerspectiveCamera.h"
#include "OptiXRenderer.h"
#include "OptiXScene.h"

namespace brayns
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
    Property stereoProperty{"stereo", isStereo, {"Stereo"}};
    Property fovy{"fovy", 45., .1, 360., {"Field of view"}};
    Property aspect{"aspect", 1., {"Aspect ratio"}};
    aspect.markReadOnly();
    Property eyeSeparation{"interpupillaryDistance", 0.0635, {"Eye separation"}};
    Property enableClippingPlanes{"enableClippingPlanes", true, {"Clipping"}};

    OptiXContext& context = OptiXContext::get();

    {
        PropertyMap properties;
        properties.setProperty(fovy);
        properties.setProperty(aspect);
        properties.setProperty({"apertureRadius", 0., {"Aperture radius"}});
        properties.setProperty({"focusDistance", 1., {"Focus Distance"}});
        properties.setProperty({"nearClip", 0., 0., 1e6, {"Near clip"}});
        properties.setProperty(enableClippingPlanes);
        properties.setProperty(stereoProperty);
        properties.setProperty(eyeSeparation);
        auto camera = std::make_shared<OptiXPerspectiveCamera>();
        context.addCamera("perspective", camera);
        addCameraType("perspective", properties);
    }

    {
        PropertyMap properties;
        properties.setProperty({"height", 1., {"Height"}});

        auto camera = std::make_shared<OptiXOrthographicCamera>();
        context.addCamera("orthographic", camera);
        addCameraType("orthographic", properties);
    }

    {
        PropertyMap properties;
        properties.setProperty({"segmentId", 7});
        properties.setProperty({"interpupillaryDistance", 0.065, {"Eye separation"}});
        properties.setProperty({"headPosition", std::array<double, 3>{{0.0, 2.0, 0.0}}});
        properties.setProperty({"headRotation", std::array<double, 4>{{0.0, 0.0, 0.0, 1.0}}});
        if (isStereo)
        {
            properties.setProperty(stereoProperty);
            properties.setProperty(eyeSeparation);
            properties.setProperty({"zeroParallaxPlane", 1., {"Zero parallax plane"}});
        }

        context.addCamera("opendeck", std::make_shared<OptiXOpenDeckCamera>());
        addCameraType("opendeck", properties);
    }
}

void OptiXEngine::_createRenderers()
{
    _renderer = std::make_shared<OptiXRenderer>(_parametersManager.getAnimationParameters(),
                                                _parametersManager.getRenderingParameters());
    _renderer->setScene(_scene);

    { // Advanced renderer
        const std::string CUDA_ADVANCED_SIMULATION = braynsOptix6Engine_generated_BioExplorer_cu_ptx;

        OptiXContext& context = OptiXContext::get();

        auto osp = std::make_shared<OptixShaderProgram>();
        osp->closest_hit =
            context.getOptixContext()->createProgramFromPTXString(CUDA_ADVANCED_SIMULATION, "closest_hit_radiance");
        osp->closest_hit_textured =
            context.getOptixContext()->createProgramFromPTXString(CUDA_ADVANCED_SIMULATION,
                                                                  "closest_hit_radiance_textured");
        osp->any_hit =
            context.getOptixContext()->createProgramFromPTXString(CUDA_ADVANCED_SIMULATION, "any_hit_shadow");
        // Exception program
        osp->exception_program =
            context.getOptixContext()->createProgramFromPTXString(CUDA_ADVANCED_SIMULATION, "exception");
        context.getOptixContext()->setExceptionProgram(0, osp->exception_program);
        context.getOptixContext()["bad_color"]->setFloat(1.0f, 0.0f, 0.0f);

        context.addRenderer("bio_explorer", osp);

        PropertyMap properties;
        properties.setProperty({"epsilonFactor", 1.0, 1.0, 1000.0, {"Epsilon factor"}});
        properties.setProperty({"shadows", 0., 0., 1., {"Shadow strength"}});
        properties.setProperty({"softShadows", 0., 0., 1., {"Soft shadow strength"}});
        properties.setProperty({"softShadowsSamples", 1, 1, 64, {"Soft shadow samples"}});
        properties.setProperty({"giDistance", 10000.0, {"Global illumination distance"}});
        properties.setProperty({"giWeight", 0.0, 1.0, 1.0, {"Global illumination weight"}});
        properties.setProperty({"giSamples", 0, 0, 64, {"Global illumination samples"}});
        properties.setProperty({"maxBounces", 3, 1, 20, {"Max ray recursion depth"}});
        properties.setProperty({"mainExposure", 1.0, 0.01, 10.0, {"Exposure"}});
        properties.setProperty({"matrixFilter", false, {"Matrix filter"}});
        properties.setProperty({"showBackground", false, {"Show background"}});

        addRendererType("bio_explorer", properties);
    }

    { // Basic simulation / Basic renderer
        const std::string CUDA_BASIC_SIMULATION_RENDERER = braynsOptix6Engine_generated_Basic_cu_ptx;
        OptiXContext& context = OptiXContext::get();

        auto osp = std::make_shared<OptixShaderProgram>();
        osp->closest_hit = context.getOptixContext()->createProgramFromPTXString(CUDA_BASIC_SIMULATION_RENDERER,
                                                                                 "closest_hit_radiance");
        osp->closest_hit_textured =
            context.getOptixContext()->createProgramFromPTXString(CUDA_BASIC_SIMULATION_RENDERER,
                                                                  "closest_hit_radiance_textured");
        osp->any_hit =
            context.getOptixContext()->createProgramFromPTXString(CUDA_BASIC_SIMULATION_RENDERER, "any_hit_shadow");

        context.addRenderer("basic", osp);

        PropertyMap properties;
        properties.setProperty({"mainExposure", 1.0, 0.01, 10.0, {"Exposure"}});
        addRendererType("basic", properties);
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
} // namespace brayns

extern "C" brayns::Engine* brayns_engine_create(int, const char**, brayns::ParametersManager& parametersManager)
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

    return new brayns::OptiXEngine(parametersManager);
}
