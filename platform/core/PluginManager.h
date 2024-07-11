/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#pragma once

#include <platform/core/common/Types.h>
#include <platform/core/common/utils/DynamicLib.h>
#include <platform/core/pluginapi/ExtensionPlugin.h>

#include <vector>

namespace core
{
/**
 */
class PluginManager
{
public:
    /**
     * @brief Constructor
     * @param argc Number of command line arguments
     * @param argv Command line arguments
     */
    PluginManager(int argc, const char** argv);

    /** Calls ExtensionPlugin::init in all loaded plugins */
    void initPlugins(PluginAPI* api);

    /** Destroys all plugins. */
    void destroyPlugins();

    /** Calls ExtensionPlugin::preRender in all loaded plugins */
    void preRender();

    /** Calls ExtensionPlugin::postRender in all loaded plugins */
    void postRender();

private:
    std::vector<DynamicLib> _libs;
    std::vector<std::unique_ptr<ExtensionPlugin>> _extensions;

    void _loadPlugin(const char* name, int argc, const char* argv[]);
};
} // namespace core
