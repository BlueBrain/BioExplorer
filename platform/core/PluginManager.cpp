/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Juan Hernando <juan.hernando@epfl.ch>
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

#include "PluginManager.h"

#include <Defines.h>

#include <platform/core/common/Logs.h>
#include <platform/core/common/utils/stringUtils.h>
#include <platform/core/parameters/ParametersManager.h>

#include <platform/core/pluginapi/Plugin.h>
#ifdef USE_NETWORKING
#include <platform/plugins/rockets/RocketsPlugin.h>
#endif

namespace
{
bool containsString(const int length, const char** input, const char* toFind)
{
    return std::count_if(input, input + length, [toFind](const char* arg) { return std::strcmp(arg, toFind) == 0; }) >
           0;
}
} // namespace

namespace core
{
typedef ExtensionPlugin* (*CreateFuncType)(int, const char**);

PluginManager::PluginManager(int argc, const char** argv)
{
    const bool help = containsString(argc, argv, "--help");

    for (int i = 0; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--plugin") != 0)
            continue;
        if (++i == argc || argv[i][0] == '\0' || argv[i][0] == '-')
        {
            // Do not print anything here, errors will be reported later
            // during option parsing
            continue;
        }

        std::string str(argv[i]);
        string_utils::trim(str);
        auto words = string_utils::split(str, ' ');

        if (help)
            words.push_back("--help");

        const char* name = words.front().c_str();
        std::vector<const char*> args;
        for (const auto& w : words)
            args.push_back(w.c_str());

        _loadPlugin(name, args.size(), args.data());
    }
}

void PluginManager::initPlugins(PluginAPI* api)
{
    // Rockets plugin cannot be initialized until we have the command line
    // parameters
    auto& parameters = api->getParametersManager();
    auto& appParameters = parameters.getApplicationParameters();

    const bool haveHttpServerURI = !appParameters.getHttpServerURI().empty();

    if (haveHttpServerURI)
#ifdef USE_NETWORKING
        // Since the Rockets plugin provides the ActionInterface, it must be
        // initialized before anything else
        _extensions.insert(_extensions.begin(), std::make_unique<RocketsPlugin>());
#else
        throw std::runtime_error(
            "CORE_NETWORKING_ENABLED was not set, but HTTP server URI "
            "was specified");
#endif

    for (const auto& extension : _extensions)
    {
        extension->_api = api;
        extension->init();
    }
}

void PluginManager::destroyPlugins()
{
    _extensions.clear();
    _libs.clear();
}

void PluginManager::preRender()
{
    for (const auto& extension : _extensions)
        extension->preRender();
}

void PluginManager::postRender()
{
    for (const auto& extension : _extensions)
        extension->postRender();
}

void PluginManager::_loadPlugin(const char* name, int argc, const char* argv[])
{
    try
    {
        DynamicLib library(name);
        auto createSym = library.getSymbolAddress("core_plugin_create");
        if (!createSym)
        {
            throw std::runtime_error(std::string("Plugin '") + name + "' is not a valid Core plugin; missing " +
                                     "core_plugin_create()");
        }

        CreateFuncType createFunc = (CreateFuncType)createSym;
        if (auto plugin = createFunc(argc, argv))
        {
            _extensions.emplace_back(plugin);
            _libs.push_back(std::move(library));
            CORE_INFO("Loaded plugin '" << name << "'");
        }
    }
    catch (const std::runtime_error& exc)
    {
        CORE_ERROR(exc.what());
    }
}
} // namespace core
