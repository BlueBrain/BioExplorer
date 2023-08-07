/*
 * Copyright (c) 2018-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
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
 * along with this library; if not, write to the Free Software Foundation, Inc.,l
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "MultiviewPlugin.h"

#include <Version.h>

#include <platform/core/common/Logs.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>

const std::string RENDERER_MULTI_VIEW = "multiview";
const std::string PARAM_ARM_LENGTH = "armLength";
const std::string PARAM_HEIGHT = "height";

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
    properties.setProperty(
        {PARAM_HEIGHT, 10.0, 0.0, 100.0, {"View height", "The height of the viewport in world space"}});

    if (!properties.parse(argc, argv))
        return nullptr;
    return new core::MultiviewPlugin(std::move(properties));
}
