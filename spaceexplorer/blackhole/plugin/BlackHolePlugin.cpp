/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

#include "BlackHolePlugin.h"

#include <Version.h>

#include <plugin/common/Logs.h>

#include <platform/core/common/ActionInterface.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>

namespace spaceexplorer
{
namespace blackhole
{
using namespace core;

const std::string PLUGIN_API_PREFIX = "bh_";

#define CATCH_STD_EXCEPTION()           \
    catch (const std::runtime_error &e) \
    {                                   \
        response.status = false;        \
        response.contents = e.what();   \
        PLUGIN_ERROR << e.what());      \
        ;                               \
    }

void _addBlackHoleRenderer(Engine &engine)
{
    PLUGIN_INFO("Registering 'black_hole_renderer' renderer");
    PropertyMap properties;
    properties.setProperty({"exposure", 1., 1., 10., {"Exposure"}});
    properties.setProperty({"nbDisks", 20, 2, 128, {"Number of disks"}});
    properties.setProperty({"grid", false, {"Display grid"}});
    properties.setProperty({"timestamp", 0., 0., 8192., {"Timestamp"}});
    properties.setProperty({"diskRotationSpeed", 3., 1., 10., {"Disk rotation speed"}});
    properties.setProperty({"diskTextureLayers", 12, 2, 100, {"Disk texture layers"}});
    properties.setProperty({"blackHoleSize", 0.3, 0.1, 10., {"Black hole size"}});
    engine.addRendererType("black_hole_renderer", properties);
}

BlackHolePlugin::BlackHolePlugin()
    : ExtensionPlugin()
{
}

void BlackHolePlugin::init()
{
    auto actionInterface = _api->getActionInterface();
    auto &engine = _api->getEngine();

    if (actionInterface)
    {
        std::string entryPoint = PLUGIN_API_PREFIX + "version";
        PLUGIN_INFO("Registering '" + entryPoint + "' endpoint");
        actionInterface->registerRequest<Response>(entryPoint, [&]() { return _version(); });
    }

    _addBlackHoleRenderer(engine);
    _api->getParametersManager().getRenderingParameters().setCurrentRenderer("black_hole_renderer");
}

Response BlackHolePlugin::_version() const
{
    Response response;
    response.contents = PACKAGE_VERSION_STRING;
    return response;
}

extern "C" ExtensionPlugin *core_plugin_create(int argc, char **argv)
{
    PLUGIN_INFO("");
    PLUGIN_INFO("_|_|_|    _|                      _|            _|    _|            _|            ");
    PLUGIN_INFO("_|    _|  _|    _|_|_|    _|_|_|  _|  _|        _|    _|    _|_|    _|    _|_|    ");
    PLUGIN_INFO("_|_|_|    _|  _|    _|  _|        _|_|          _|_|_|_|  _|    _|  _|  _|_|_|_|  ");
    PLUGIN_INFO("_|    _|  _|  _|    _|  _|        _|  _|        _|    _|  _|    _|  _|  _|        ");
    PLUGIN_INFO("_|_|_|    _|    _|_|_|    _|_|_|  _|    _|      _|    _|    _|_|    _|    _|_|_|  ");
    PLUGIN_INFO("");

    return new BlackHolePlugin();
}

} // namespace blackhole
} // namespace spaceexplorer