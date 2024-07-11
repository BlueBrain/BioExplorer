/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

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

#include "OpenDeckPlugin.h"

#include <plugin/common/Logs.h>

#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>

#ifdef USE_OPTIX6
#include <platform/engines/optix6/OptiXContext.h>
#include <plugin/optix6/OptiXCylindricStereoCamera.h>
#endif

namespace core
{
namespace
{
const std::string CAMERA_CYLINDRIC = "cylindric";
const std::string CAMERA_CYLINDRIC_STEREO = "cylindricStereo";
const std::string CAMERA_CYLINDRIC_STEREO_TRACKED = "cylindricStereoTracked";

constexpr uint32_t openDeckWallResX = 11940u;
constexpr uint32_t openDeckWallResY = 3424u;
constexpr uint32_t openDeckFloorResX = 4096u;
constexpr uint32_t openDeckFloorResY = 2125u;

constexpr char leftWallBufferName[] = "0L";
constexpr char rightWallBufferName[] = "0R";
constexpr char leftFloorBufferName[] = "1L";
constexpr char rightFloorBufferName[] = "1R";

const std::string HEAD_POSITION_PROP = "headPosition";
const std::string HEAD_ROTATION_PROP = "headRotation";

constexpr std::array<double, 3> HEAD_INIT_POS{{0.0, 2.0, 0.0}};
constexpr std::array<double, 4> HEAD_INIT_ROT{{0.0, 0.0, 0.0, 1.0}};

Property getHeadPositionProperty()
{
    Property headPosition{HEAD_POSITION_PROP, HEAD_INIT_POS};
    headPosition.markReadOnly();
    return headPosition;
}

Property getHeadRotationProperty()
{
    Property headRotation{HEAD_ROTATION_PROP, HEAD_INIT_ROT};
    headRotation.markReadOnly();
    return headRotation;
}

Property getStereoModeProperty()
{
    return {"stereoMode",
            3, // side-by-side
            {"None", "Left eye", "Right eye", "Side by side"},
            {"Stereo mode"}};
}

Property getInterpupillaryDistanceProperty()
{
    return CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE;
}

Property getCameraScalingProperty(const double scaling)
{
    return {PARAM_CAMERA_SCALING, scaling, {"Camera scaling"}};
}

PropertyMap getCylindricStereoProperties()
{
    PropertyMap properties;
    properties.setProperty(getStereoModeProperty());
    properties.setProperty(getInterpupillaryDistanceProperty());
    return properties;
}

PropertyMap getCylindricStereoTrackedProperties(const OpenDeckParameters& params)
{
    PropertyMap properties;
    properties.setProperty(getHeadPositionProperty());
    properties.setProperty(getHeadRotationProperty());
    properties.setProperty(getStereoModeProperty());
    properties.setProperty(getInterpupillaryDistanceProperty());
    properties.setProperty(getCameraScalingProperty(params.getCameraScaling()));
    return properties;
}
} // namespace

OpenDeckPlugin::OpenDeckPlugin(const OpenDeckParameters& params)
    : _params(params)
{
    if (_params.getResolutionScaling() > 1.0f || _params.getResolutionScaling() <= 0.0f)
    {
        throw std::runtime_error(
            "The scale of the native OpenDeck resolution cannot be bigger "
            "than 1.0, zero or negative.");
    }
    if (_params.getCameraScaling() <= 0.0)
        throw std::runtime_error("The camera scale cannot be zero or negative");

    _wallRes =
        Vector2ui(openDeckWallResX * _params.getResolutionScaling(), openDeckWallResY * _params.getResolutionScaling());
    _floorRes = Vector2ui(openDeckFloorResX * _params.getResolutionScaling(),
                          openDeckFloorResY * _params.getResolutionScaling());
    PLUGIN_INFO("Wall resolution : " << _wallRes << "(" << _params.getResolutionScaling() << ")");
    PLUGIN_INFO("Floor resolution: " << _floorRes);
}

void OpenDeckPlugin::init()
{
    auto& engine = _api->getEngine();
    auto& params = engine.getParametersManager().getApplicationParameters();
    const auto& engineName = params.getEngine();
#ifdef USE_OSPRAY
    if (engineName == ENGINE_OSPRAY)
    {
        engine.addCameraType(CAMERA_CYLINDRIC);
        engine.addCameraType(CAMERA_CYLINDRIC_STEREO, getCylindricStereoProperties());
        engine.addCameraType(CAMERA_CYLINDRIC_STEREO_TRACKED, getCylindricStereoTrackedProperties(_params));
        FrameBufferPtr frameBuffer = engine.createFrameBuffer(leftWallBufferName, _wallRes, FrameBufferFormat::rgba_i8);
        engine.addFrameBuffer(frameBuffer);
        frameBuffer = engine.createFrameBuffer(rightWallBufferName, _wallRes, FrameBufferFormat::rgba_i8);
        engine.addFrameBuffer(frameBuffer);
        frameBuffer = engine.createFrameBuffer(leftFloorBufferName, _floorRes, FrameBufferFormat::rgba_i8);
        engine.addFrameBuffer(frameBuffer);
        frameBuffer = engine.createFrameBuffer(rightFloorBufferName, _floorRes, FrameBufferFormat::rgba_i8);
        engine.addFrameBuffer(frameBuffer);
    }
#endif
#ifdef USE_OPTIX6
    if (engineName == ENGINE_OPTIX_6)
    {
        core::engine::optix::OptiXContext& context = core::engine::optix::OptiXContext::get();
        context.addCamera(CAMERA_CYLINDRIC_STEREO, std::make_shared<core::engine::optix::OptiXCylindricStereoCamera>());
        engine.addCameraType(CAMERA_CYLINDRIC_STEREO, getCylindricStereoProperties());
    }
#endif
}

extern "C" core::ExtensionPlugin* core_plugin_create(const int argc, const char** argv)
{
    PLUGIN_INFO("");
    PLUGIN_INFO("   _|_|                                  _|_|_|                        _|      ");
    PLUGIN_INFO(" _|    _|  _|_|_|      _|_|    _|_|_|    _|    _|    _|_|      _|_|_|  _|  _|  ");
    PLUGIN_INFO(" _|    _|  _|    _|  _|_|_|_|  _|    _|  _|    _|  _|_|_|_|  _|        _|_|    ");
    PLUGIN_INFO(" _|    _|  _|    _|  _|        _|    _|  _|    _|  _|        _|        _|  _|  ");
    PLUGIN_INFO("   _|_|    _|_|_|      _|_|_|  _|    _|  _|_|_|      _|_|_|    _|_|_|  _|    _|");
    PLUGIN_INFO("           _|                                                                  ");
    PLUGIN_INFO("           _|                                                                  ");
    PLUGIN_INFO("");

    core::OpenDeckParameters params;
    if (!params.getPropertyMap().parse(argc, argv))
        return nullptr;
    try
    {
        return new core::OpenDeckPlugin(std::move(params));
    }
    catch (const std::runtime_error& exc)
    {
        PLUGIN_ERROR(exc.what());
        return nullptr;
    }
}
} // namespace core
