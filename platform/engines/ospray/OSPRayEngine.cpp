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

#include "OSPRayEngine.h"

#include <platform/core/common/Properties.h>
#include <platform/core/common/input/KeyboardHandler.h>

#include <platform/core/parameters/ParametersManager.h>

#include "Logs.h"
#include "OSPRayCamera.h"
#include "OSPRayFrameBuffer.h"
#include "OSPRayMaterial.h"
#include "OSPRayProperties.h"
#include "OSPRayRenderer.h"
#include "OSPRayScene.h"

#include <ospray/OSPConfig.h>                    // TILE_SIZE
#include <ospray/SDK/camera/PerspectiveCamera.h> // enum StereoMode
#include <ospray/version.h>

namespace core
{
namespace engine
{
namespace ospray
{
OSPRayEngine::OSPRayEngine(ParametersManager& parametersManager)
    : Engine(parametersManager)
{
    auto& ap = _parametersManager.getApplicationParameters();
    try
    {
        std::vector<const char*> argv;

        // Ospray expects but ignores the application name as the first argument
        argv.push_back("Core");

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
        CORE_ERROR("Error during ospInit(): " << e.what());
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
            CORE_ERROR("Error while loading module " << module << ": " << e.what());
        }
    }

    _createRenderers();

    _scene = std::make_shared<OSPRayScene>(_parametersManager.getAnimationParameters(),
                                           _parametersManager.getGeometryParameters(),
                                           _parametersManager.getVolumeParameters(),
                                           _parametersManager.getFieldParameters());

    _createCameras();

    _renderer->setEngine(this);
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
            ospDeviceSet1i(device, OSPRAY_ENGINE_PROPERTY_LOAD_BALANCER_DYNAMIC, useDynamicLoadBalancer);
            ospDeviceCommit(device);
            _useDynamicLoadBalancer = useDynamicLoadBalancer;

            PLUGIN_INFO("Using " << (useDynamicLoadBalancer ? "dynamic" : "static") << " load balancer");
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
        PLUGIN_INFO("Registering '" << RENDERER_PROPERTY_TYPE_ADVANCED << "' renderer");
        PropertyMap properties;
        properties.setProperty(RENDERER_PROPERTY_FAST_PREVIEW);
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
        properties.setProperty(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER);
        properties.setProperty(RENDERER_PROPERTY_SHOW_BACKGROUND);
        properties.setProperty(RENDERER_PROPERTY_MATRIX_FILTER);
        properties.setProperty(OSPRAY_RENDERER_VOLUME_SAMPLING_THRESHOLD);
        properties.setProperty(OSPRAY_RENDERER_VOLUME_SPECULAR_EXPONENT);
        properties.setProperty(OSPRAY_RENDERER_VOLUME_ALPHA_CORRECTION);
        addRendererType(RENDERER_PROPERTY_TYPE_ADVANCED, properties);
    }
    {
        PLUGIN_INFO("Registering '" << RENDERER_PROPERTY_TYPE_SCIVIS << "' renderer");
        PropertyMap properties;
        properties.setProperty(OSPRAY_RENDERER_AMBIENT_OCCLUSION_DISTANCE);
        properties.setProperty(OSPRAY_RENDERER_AMBIENT_OCCLUSION_SAMPLES);
        properties.setProperty(OSPRAY_RENDERER_AMBIENT_OCCLUSION_ENABLED);
        properties.setProperty(OSPRAY_RENDERER_AMBIENT_OCCLUSION_WEIGHT);
        properties.setProperty(OSPRAY_RENDERER_ONE_SIDED_LIGHTING);
        properties.setProperty(OSPRAY_RENDERER_SHADOW_ENABLED);
        addRendererType(RENDERER_PROPERTY_TYPE_SCIVIS, properties);
    }
    {
        PLUGIN_INFO("Registering '" << RENDERER_PROPERTY_TYPE_BASIC << "' renderer");
        addRendererType(RENDERER_PROPERTY_TYPE_BASIC);
    }
}

FrameBufferPtr OSPRayEngine::createFrameBuffer(const std::string& name, const Vector2ui& frameSize,
                                               const FrameBufferFormat frameBufferFormat) const
{
    return std::make_shared<OSPRayFrameBuffer>(name, frameSize, frameBufferFormat);
}

ScenePtr OSPRayEngine::createScene(AnimationParameters& animationParameters, GeometryParameters& geometryParameters,
                                   VolumeParameters& volumeParameters, FieldParameters& fieldParameters) const
{
    return std::make_shared<OSPRayScene>(animationParameters, geometryParameters, volumeParameters, fieldParameters);
}

CameraPtr OSPRayEngine::createCamera() const
{
    return std::make_shared<OSPRayCamera>(const_cast<OSPRayEngine*>(this));
}

RendererPtr OSPRayEngine::createRenderer(const AnimationParameters& animationParameters,
                                         const RenderingParameters& renderingParameters) const
{
    return std::make_shared<OSPRayRenderer>(animationParameters, renderingParameters);
}

void OSPRayEngine::_createCameras()
{
    _camera = std::make_shared<OSPRayCamera>(this);

    const bool isStereo = _parametersManager.getApplicationParameters().isStereo();
    Property stereoProperty{CAMERA_PROPERTY_STEREO.name, isStereo, {CAMERA_PROPERTY_STEREO.metaData}};
    Property aspect = CAMERA_PROPERTY_ASPECT_RATIO;
    aspect.markReadOnly();
    {
        PropertyMap properties;
        properties.setProperty(CAMERA_PROPERTY_FIELD_OF_VIEW);
        properties.setProperty(aspect);
        properties.setProperty(CAMERA_PROPERTY_APERTURE_RADIUS);
        properties.setProperty(CAMERA_PROPERTY_FOCAL_DISTANCE);
        properties.setProperty(CAMERA_PROPERTY_NEAR_CLIP);
        properties.setProperty(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES);
        properties.setProperty(stereoProperty);
        properties.setProperty(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE);
        properties.setProperty(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER);
        addCameraType(CAMERA_PROPERTY_TYPE_PERSPECTIVE, properties);
    }
    {
        PropertyMap properties;
        properties.setProperty(CAMERA_PROPERTY_HEIGHT);
        properties.setProperty(aspect);
        properties.setProperty(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES);
        addCameraType(CAMERA_PROPERTY_TYPE_ORTHOGRAPHIC, properties);
    }
    {
        PropertyMap properties;
        properties.setProperty(CAMERA_PROPERTY_FIELD_OF_VIEW);
        properties.setProperty(aspect);
        properties.setProperty(CAMERA_PROPERTY_APERTURE_RADIUS);
        properties.setProperty(CAMERA_PROPERTY_FOCAL_DISTANCE);
        properties.setProperty(CAMERA_PROPERTY_NEAR_CLIP);
        properties.setProperty(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES);
        properties.setProperty(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE);
        properties.setProperty(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER);
        addCameraType(CAMERA_PROPERTY_TYPE_ANAGLYPH, properties);
    }
    {
        PropertyMap properties;
        properties.setProperty(CAMERA_PROPERTY_FIELD_OF_VIEW);
        properties.setProperty(aspect);
        properties.setProperty(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES);
        if (isStereo)
        {
            properties.setProperty(stereoProperty);
            properties.setProperty(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE);
            properties.setProperty(OSPRAY_CAMERA_PROPERTY_ZERO_PARALLAX_PLANE);
        }
        addCameraType(OSPRAY_CAMERA_PROPERTY_TYPE_PERSPECTIVE_PARALLAX, properties);
    }
    {
        PropertyMap properties;
        properties.setProperty(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES);
        properties.setProperty(OSPRAY_CAMERA_PROPERTY_HALF_SPHERE);
        if (isStereo)
        {
            properties.setProperty(stereoProperty);
            properties.setProperty(CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE);
        }
        addCameraType(OSPRAY_CAMERA_PROPERTY_TYPE_PANORAMIC, properties);
    }
    {
        PropertyMap properties;
        properties.setProperty(CAMERA_PROPERTY_FIELD_OF_VIEW);
        properties.setProperty(aspect);
        properties.setProperty(CAMERA_PROPERTY_APERTURE_RADIUS);
        properties.setProperty(CAMERA_PROPERTY_FOCAL_DISTANCE);
        properties.setProperty(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES);
        properties.setProperty(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER);
        addCameraType(OSPRAY_CAMERA_PROPERTY_TYPE_FISHEYE, properties);
    }
}
} // namespace ospray
} // namespace engine
} // namespace core

extern "C" core::Engine* core_engine_create(int, const char**, core::ParametersManager& parametersManager)
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

    return new core::engine::ospray::OSPRayEngine(parametersManager);
}
