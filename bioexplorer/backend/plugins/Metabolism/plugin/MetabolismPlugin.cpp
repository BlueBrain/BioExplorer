/*
 * Copyright 2020-2024 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "MetabolismPlugin.h"

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

#include <platform/core/common/ActionInterface.h>
#include <platform/core/common/Properties.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>

#include <fstream>

using namespace core;

namespace bioexplorer
{
using namespace details;
namespace metabolism
{

const std::string PLUGIN_VERSION = "0.1.0";
const std::string PLUGIN_API_PREFIX = "mb-";

const std::string RENDERER_METABOLISM = "metabolism";

#define CATCH_STD_EXCEPTION()                  \
    catch (const std::runtime_error &e)        \
    {                                          \
        response.status = false;               \
        response.contents = e.what();          \
        PLUGIN_ERROR << e.what() << std::endl; \
    }

void _addMetabolismRenderer(Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_METABOLISM);
    PropertyMap properties;
    properties.setProperty(COMMON_PROPERTY_EXPOSURE);
    properties.setProperty({"rayStep", 0.1, 0.01, 10., {"Ray marching step"}});
    properties.setProperty({"nearPlane", 10., 0.01, 1e6, {"Near plane"}});
    properties.setProperty({"farPlane", 50., 0.01, 1e6, {"Far plane"}});
    properties.setProperty({"refinementSteps", 64, 1, 256, {"Refinement steps"}});
    properties.setProperty(RENDERER_PROPERTY_ALPHA_CORRECTION);
    properties.setProperty({"noiseFrequency", 1., 0., 100., {"Noise frequency"}});
    properties.setProperty({"noiseAmplitude", 1., 0.00001, 10., {"Noise amplitude"}});
    properties.setProperty({"colorMapPerRegion", true, {"Color map per region"}});
    engine.addRendererType(RENDERER_METABOLISM, properties);
}

MetabolismPlugin::MetabolismPlugin(int argc, char **argv)
    : ExtensionPlugin()
{
    _parseCommandLineArguments(argc, argv);
}

void MetabolismPlugin::init()
{
    auto actionInterface = _api->getActionInterface();
    auto &engine = _api->getEngine();

    if (actionInterface)
    {
        std::string endPoint = PLUGIN_API_PREFIX + "attach-handler";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AttachHandlerDetails, Response>(endPoint,
                                                                         [&](const AttachHandlerDetails &payload)
                                                                         { return _attachHandler(payload); });
    }

    _addMetabolismRenderer(engine);
}

void MetabolismPlugin::_parseCommandLineArguments(int argc, char **argv)
{
    for (size_t i = 0; i < argc; ++i)
    {
        const std::string argument = argv[i];
        std::string key;
        std::string value;
        const int pos = argument.find("=");
        if (pos == std::string::npos)
            key = argument;
        else
        {
            key = argument.substr(0, pos);
            value = argument.substr(pos + 1);
        }
        _commandLineArguments[key] = value;
    }
}

Response MetabolismPlugin::_attachHandler(const AttachHandlerDetails &payload)
{
    Response response;
    try
    {
        auto &scene = _api->getScene();
        auto descriptors = scene.getModelDescriptors();
        if (descriptors.empty())
            PLUGIN_THROW("Scene must contain a model");

        auto descriptor = descriptors[0];
        auto &model = descriptor->getModel();

        auto handler = std::make_shared<MetabolismHandler>(payload);
        model.setSimulationHandler(handler);
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
        PLUGIN_ERROR(e.what());
    }
    return response;
}

extern "C" ExtensionPlugin *core_plugin_create(int argc, char **argv)
{
    PLUGIN_INFO("Initializing Metabolism plug-in (version " << PLUGIN_VERSION << ")");
    PLUGIN_INFO("");
    PLUGIN_INFO("_|      _|              _|                _|                  _|  _|                            ");
    PLUGIN_INFO("_|_|  _|_|    _|_|    _|_|_|_|    _|_|_|  _|_|_|      _|_|    _|        _|_|_|  _|_|_|  _|_|    ");
    PLUGIN_INFO("_|  _|  _|  _|_|_|_|    _|      _|    _|  _|    _|  _|    _|  _|  _|  _|_|      _|    _|    _|  ");
    PLUGIN_INFO("_|      _|  _|          _|      _|    _|  _|    _|  _|    _|  _|  _|      _|_|  _|    _|    _|  ");
    PLUGIN_INFO("_|      _|    _|_|_|      _|_|    _|_|_|  _|_|_|      _|_|    _|  _|  _|_|_|    _|    _|    _|  ");
    PLUGIN_INFO("");

    return new MetabolismPlugin(argc, argv);
}

} // namespace metabolism
} // namespace bioexplorer
