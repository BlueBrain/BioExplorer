/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue Brain Project / EPFL
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

#include <brayns/common/ActionInterface.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/parameters/ParametersManager.h>
#include <brayns/pluginapi/Plugin.h>

#include <fstream>

namespace bioexplorer
{
namespace metabolism
{
using namespace bioexplorer;
using namespace details;

const std::string PLUGIN_VERSION = "0.1.0";
const std::string PLUGIN_API_PREFIX = "mb-";

#define CATCH_STD_EXCEPTION()                  \
    catch (const std::runtime_error &e)        \
    {                                          \
        response.status = false;               \
        response.contents = e.what();          \
        PLUGIN_ERROR << e.what() << std::endl; \
    }

void _addMetabolismRenderer(Engine &engine)
{
    PLUGIN_INFO("Registering 'metabolism' renderer");
    PropertyMap properties;
    properties.setProperty({"exposure", 1., 1., 10., {"Exposure"}});
    properties.setProperty({"rayStep", 0.1, 0.01, 10., {"Ray marching step"}});
    properties.setProperty({"nearPlane", 10., 0.01, 1e6, {"Near plane"}});
    properties.setProperty({"farPlane", 50., 0.01, 1e6, {"Far plane"}});
    properties.setProperty(
        {"refinementSteps", 64, 1, 256, {"Refinement steps"}});
    properties.setProperty(
        {"alphaCorrection", 1., 0.001, 1., {"Alpha correction"}});
    properties.setProperty(
        {"searchLength", 1., 0.001, 100., {"Search length"}});
    properties.setProperty(
        {"noiseFrequency", 1., 0., 100., {"Noise frequency"}});
    properties.setProperty(
        {"noiseAmplitude", 1., 0.00001, 10., {"Noise amplitude"}});
    properties.setProperty({"useRandomSearch", false, {"Use random search"}});
    engine.addRendererType("metabolism", properties);
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
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<AttachHandlerDetails, Response>(
            endPoint, [&](const AttachHandlerDetails &payload) {
                return _attachHandler(payload);
            });
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

extern "C" ExtensionPlugin *brayns_plugin_create(int argc, char **argv)
{
    PLUGIN_INFO("Initializing Metabolism plug-in (version " << PLUGIN_VERSION
                                                            << ")");
    PLUGIN_INFO("");
    PLUGIN_INFO(
        "_|      _|              _|                _|                  _|  _|  "
        "                          ");
    PLUGIN_INFO(
        "_|_|  _|_|    _|_|    _|_|_|_|    _|_|_|  _|_|_|      _|_|    _|      "
        "  _|_|_|  _|_|_|  _|_|    ");
    PLUGIN_INFO(
        "_|  _|  _|  _|_|_|_|    _|      _|    _|  _|    _|  _|    _|  _|  _|  "
        "_|_|      _|    _|    _|  ");
    PLUGIN_INFO(
        "_|      _|  _|          _|      _|    _|  _|    _|  _|    _|  _|  _|  "
        "    _|_|  _|    _|    _|  ");
    PLUGIN_INFO(
        "_|      _|    _|_|_|      _|_|    _|_|_|  _|_|_|      _|_|    _|  _|  "
        "_|_|_|    _|    _|    _|  ");
    PLUGIN_INFO("");

    return new MetabolismPlugin(argc, argv);
}

} // namespace metabolism
} // namespace bioexplorer
