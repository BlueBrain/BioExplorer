/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
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
