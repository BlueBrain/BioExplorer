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

#include "MultiviewPlugin.h"

#include <Version.h>

#include <platform/core/common/Logs.h>
#include <platform/core/common/Properties.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>
#include <platform/plugins/multiview/common/CommonStructs.h>

const std::string RENDERER_MULTI_VIEW = "multiview";

namespace core
{
MultiviewPlugin::MultiviewPlugin(PropertyMap&& properties)
    : _properties(std::move(properties))
{
    const double armLength = _properties.getProperty<double>(PARAM_ARM_LENGTH);
    if (armLength <= 0.0f)
        CORE_THROW("The " + RENDERER_MULTI_VIEW + " camera arm length must be strictly positive");
}

void MultiviewPlugin::init()
{
    auto& engine = _api->getEngine();
    auto& params = engine.getParametersManager();
    if (params.getApplicationParameters().getEngine() == ENGINE_OSPRAY)
    {
        PLUGIN_REGISTER_CAMERA(RENDERER_MULTI_VIEW);
        engine.addCameraType(RENDERER_MULTI_VIEW, _properties);
    }
    else
        CORE_THROW("The " + RENDERER_MULTI_VIEW + " camera is only available for " + ENGINE_OSPRAY + " engine");
}
} // namespace core

extern "C" core::ExtensionPlugin* core_plugin_create(const int argc, const char** argv)
{
    CORE_INFO("");
    CORE_INFO(" _|      _|            _|    _|      _|                          _|                              ");
    CORE_INFO(" _|_|  _|_|  _|    _|  _|  _|_|_|_|                  _|      _|        _|_|    _|      _|      _|");
    CORE_INFO(" _|  _|  _|  _|    _|  _|    _|      _|  _|_|_|_|_|  _|      _|  _|  _|_|_|_|  _|      _|      _|");
    CORE_INFO(" _|      _|  _|    _|  _|    _|      _|                _|  _|    _|  _|          _|  _|  _|  _|  ");
    CORE_INFO(" _|      _|    _|_|_|  _|      _|_|  _|                  _|      _|    _|_|_|      _|      _|    ");
    CORE_INFO("");
    CORE_INFO("Initializing Multi-view plug-in (version " << PACKAGE_VERSION_STRING << ")");

    core::PropertyMap properties;
    properties.setProperty(
        {PARAM_ARM_LENGTH, 5.0, 0.0, 100.0, {"Arm length", "The distance between the cameras and the view center"}});
    properties.setProperty({core::CAMERA_PROPERTY_HEIGHT.name,
                            10.0,
                            0.0,
                            100.0,
                            {"View height", "The height of the viewport in world space"}});

    if (!properties.parse(argc, argv))
        return nullptr;
    return new core::MultiviewPlugin(std::move(properties));
}
