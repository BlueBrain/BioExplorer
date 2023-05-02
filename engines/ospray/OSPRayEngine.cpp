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

#include "OSPRayEngine.h"

#include <brayns/common/input/KeyboardHandler.h>

#include <brayns/parameters/ParametersManager.h>

#include "Logs.h"
#include "OSPRayCamera.h"
#include "OSPRayFrameBuffer.h"
#include "OSPRayMaterial.h"
#include "OSPRayRenderer.h"
#include "OSPRayScene.h"

#include <ospray/OSPConfig.h>                    // TILE_SIZE
#include <ospray/SDK/camera/PerspectiveCamera.h> // enum StereoMode
#include <ospray/version.h>

namespace brayns
{
OSPRayEngine::OSPRayEngine(ParametersManager& parametersManager)
    : Engine(parametersManager)
{
    auto& ap = _parametersManager.getApplicationParameters();
    try
    {
        std::vector<const char*> argv;

        // Ospray expects but ignores the application name as the first argument
        argv.push_back("Brayns");

        // Setup log and error output
        argv.push_back("--osp:logoutput");
        argv.push_back("cout");
        argv.push_back("--osp:erroroutput");
        argv.push_back("cerr");

        if (_parametersManager.getApplicationParameters().getParallelRendering())
        {
            argv.push_back("--osp:mpi");
        }

        int argc = argv.size();
        ospInit(&argc, argv.data());
    }
    catch (const std::exception& e)
    {
        // Note: This is necessary because OSPRay does not yet implement a
        // ospDestroy API.
        BRAYNS_ERROR("Error during ospInit(): " << e.what());
    }

    for (const auto& module : ap.getOsprayModules())
    {
        try
        {
            const auto error = ospLoadModule(module.c_str());
            if (error != OSP_NO_ERROR)
                throw std::runtime_error(ospDeviceGetLastErrorMsg(ospGetCurrentDevice()));
        }
        catch (const std::exception& e)
        {
            BRAYNS_ERROR("Error while loading module " << module << ": " << e.what());
        }
    }

    _createRenderers();

    _scene = std::make_shared<OSPRayScene>(_parametersManager.getAnimationParameters(),
                                           _parametersManager.getGeometryParameters(),
                                           _parametersManager.getVolumeParameters());

    _createCameras();

    _renderer->setScene(_scene);
    _renderer->setCamera(_camera);
}

OSPRayEngine::~OSPRayEngine()
{
    _scene.reset();
    _frameBuffers.clear();
    _renderer.reset();
    _camera.reset();

    ospShutdown();
}

void OSPRayEngine::commit()
{
    Engine::commit();

    auto device = ospGetCurrentDevice();
    if (device && _parametersManager.getRenderingParameters().isModified())
    {
        const auto useDynamicLoadBalancer = _parametersManager.getApplicationParameters().getDynamicLoadBalancer();
        if (_useDynamicLoadBalancer != useDynamicLoadBalancer)
        {
            ospDeviceSet1i(device, "dynamicLoadBalancer", useDynamicLoadBalancer);
            ospDeviceCommit(device);
            _useDynamicLoadBalancer = useDynamicLoadBalancer;

            BRAYNS_INFO("Using " << (useDynamicLoadBalancer ? "dynamic" : "static") << " load balancer");
        }
    }
}

Vector2ui OSPRayEngine::getMinimumFrameSize() const
{
    return {TILE_SIZE, TILE_SIZE};
}

void OSPRayEngine::_createRenderers()
{
    _renderer = std::make_shared<OSPRayRenderer>(_parametersManager.getAnimationParameters(),
                                                 _parametersManager.getRenderingParameters());

    {
        BRAYNS_INFO("Registering 'advanced' renderer");
        PropertyMap properties;
        properties.setProperty({"alphaCorrection", 0.5, 0.001, 1., {"Alpha correction"}});
        properties.setProperty(
            {"maxDistanceToSecondaryModel", 30., 0.1, 100., {"Maximum distance to secondary model"}});
        properties.setProperty({"giDistance", 10000.0, {"Global illumination distance"}});
        properties.setProperty({"giWeight", 0.0, 1.0, 1.0, {"Global illumination weight"}});
        properties.setProperty({"giSamples", 0, 0, 64, {"Global illumination samples"}});
        properties.setProperty({"shadows", 0.0, 0.0, 1.0, {"Shadow intensity"}});
        properties.setProperty({"softShadows", 0.0, 0.0, 1.0, {"Shadow softness"}});
        properties.setProperty({"softShadowsSamples", 1, 1, 64, {"Soft shadow samples"}});
        properties.setProperty({"mainExposure", 1.0, 0.01, 10.0, {"Exposure"}});
        properties.setProperty({"epsilonFactor", 1.0, 1.0, 1000.0, {"Epsilon factor"}});
        properties.setProperty({"fogStart", 0.0, 0.0, 1e6, {"Fog start"}});
        properties.setProperty({"fogThickness", 1e6, 1e6, 1e6, {"Fog thickness"}});
        properties.setProperty({"maxBounces", 3, 1, 100, {"Maximum number of ray bounces"}});
        properties.setProperty({"useHardwareRandomizer", false, {"Use hardware accelerated randomizer"}});
        properties.setProperty({"showBackground", true, {"Show background"}});
        properties.setProperty({"matrixFilter", false, {"Matrix filter"}});
        addRendererType("advanced", properties);
    }
    {
        BRAYNS_INFO("Registering 'scivis' renderer");
        PropertyMap properties;
        properties.setProperty({"aoDistance", 10000., {"Ambient occlusion distance"}});
        properties.setProperty({"aoSamples", int32_t(1), int32_t(0), int32_t(128), {"Ambient occlusion samples"}});
        properties.setProperty({"aoTransparencyEnabled", true, {"Ambient occlusion transparency"}});
        properties.setProperty({"aoWeight", 0., 0., 1., {"Ambient occlusion weight"}});
        properties.setProperty({"oneSidedLighting", true, {"One-sided lighting"}});
        properties.setProperty({"shadowsEnabled", false, {"Shadows"}});

        addRendererType("scivis", properties);
    }
    BRAYNS_INFO("Registering 'basic' renderer");
    addRendererType("basic");
}

FrameBufferPtr OSPRayEngine::createFrameBuffer(const std::string& name, const Vector2ui& frameSize,
                                               const FrameBufferFormat frameBufferFormat) const
{
    return std::make_shared<OSPRayFrameBuffer>(name, frameSize, frameBufferFormat);
}

ScenePtr OSPRayEngine::createScene(AnimationParameters& animationParameters, GeometryParameters& geometryParameters,
                                   VolumeParameters& volumeParameters) const
{
    return std::make_shared<OSPRayScene>(animationParameters, geometryParameters, volumeParameters);
}

CameraPtr OSPRayEngine::createCamera() const
{
    return std::make_shared<OSPRayCamera>();
}

RendererPtr OSPRayEngine::createRenderer(const AnimationParameters& animationParameters,
                                         const RenderingParameters& renderingParameters) const
{
    return std::make_shared<OSPRayRenderer>(animationParameters, renderingParameters);
}

void OSPRayEngine::_createCameras()
{
    _camera = std::make_shared<OSPRayCamera>();

    const bool isStereo = _parametersManager.getApplicationParameters().isStereo();
    Property stereoProperty{"stereo", isStereo, {"Stereo"}};
    Property fovy{"fovy", 45., .1, 360., {"Field of view"}};
    Property aspect{"aspect", 1., {"Aspect ratio"}};
    aspect.markReadOnly();
    Property eyeSeparation{"interpupillaryDistance", 0.0635, {"Eye separation"}};
    Property enableClippingPlanes{"enableClippingPlanes", true, {"Clipping"}};

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
        addCameraType("perspective", properties);
    }
    {
        PropertyMap properties;
        properties.setProperty({"height", 1., {"Height"}});
        properties.setProperty(aspect);
        properties.setProperty(enableClippingPlanes);
        addCameraType("orthographic", properties);
    }
    {
        PropertyMap properties;
        properties.setProperty(fovy);
        properties.setProperty(aspect);
        properties.setProperty(enableClippingPlanes);
        if (isStereo)
        {
            properties.setProperty(stereoProperty);
            properties.setProperty(eyeSeparation);
            properties.setProperty({"zeroParallaxPlane", 1., {"Zero parallax plane"}});
        }
        addCameraType("perspectiveParallax", properties);
    }
    {
        PropertyMap properties;
        properties.setProperty(enableClippingPlanes);
        properties.setProperty({"half", true, {"Half sphere"}});
        if (isStereo)
        {
            properties.setProperty(stereoProperty);
            properties.setProperty(eyeSeparation);
        }
        addCameraType("panoramic", properties);
    }
    {
        PropertyMap properties;
        properties.setProperty(fovy);
        properties.setProperty(aspect);
        properties.setProperty({"apertureRadius", 0., {"Aperture radius"}});
        properties.setProperty({"focusDistance", 1., {"Focus Distance"}});
        properties.setProperty(enableClippingPlanes);
        addCameraType("fisheye", properties);
    }
}
} // namespace brayns

extern "C" brayns::Engine* brayns_engine_create(int, const char**, brayns::ParametersManager& parametersManager)
{
    PLUGIN_INFO("");
    PLUGIN_INFO("   _|_|      _|_|_|  _|_|_|    _|_|_|                              _|  ");
    PLUGIN_INFO(" _|    _|  _|        _|    _|  _|    _|    _|_|_|  _|    _|      _|_|  ");
    PLUGIN_INFO(" _|    _|    _|_|    _|_|_|    _|_|_|    _|    _|  _|    _|        _|  ");
    PLUGIN_INFO(" _|    _|        _|  _|        _|    _|  _|    _|  _|    _|        _|  ");
    PLUGIN_INFO("   _|_|    _|_|_|    _|        _|    _|    _|_|_|    _|_|_|        _|  ");
    PLUGIN_INFO("                                                         _|            ");
    PLUGIN_INFO("                                                     _|_|              ");
    PLUGIN_INFO("");

    return new brayns::OSPRayEngine(parametersManager);
}
