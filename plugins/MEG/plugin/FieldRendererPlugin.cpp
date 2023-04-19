/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include "FieldRendererPlugin.h"

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

#include <brayns/common/ActionInterface.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/Model.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/parameters/ParametersManager.h>
#include <brayns/pluginapi/Plugin.h>

#include <fstream>

namespace fieldrenderer
{
const std::string PLUGIN_VERSION = "0.1.0";
const std::string PLUGIN_API_PREFIX = "fr_";

#define CATCH_STD_EXCEPTION()           \
    catch (const std::runtime_error &e) \
    {                                   \
        response.status = false;        \
        response.contents = e.what();   \
        PLUGIN_ERROR(e.what());         \
    }

void _addFieldRenderer(brayns::Engine &engine)
{
    PLUGIN_INFO("Registering 'field_renderer' renderer");
    brayns::PropertyMap properties;
    properties.setProperty({"exposure", 1., 1., 10., {"Exposure"}});
    properties.setProperty({"nbRefinementSteps", 8, 1, 128, {"Number of refinement steps"}});
    properties.setProperty({"nbRaySteps", 64, 2, 2048, {"Number of ray marching steps"}});
    properties.setProperty({"cutoff", 300.0, 0.001, 1e5, {"Cutoff distance"}});
    properties.setProperty({"spaceDistortion", 0.0, 0.0, 10.0, {"Space distortion"}});
    properties.setProperty({"alphaCorrection", 0.05, 0.001, 1.0, {"Alpha correction"}});
    properties.setProperty({"renderDirections", false, {"Render field directions"}});
    properties.setProperty({"processGeometry", false, {"Include geometry (slow)"}});
    properties.setProperty({"normalized", false, {"Nomalize field values"}});
    engine.addRendererType("field_renderer", properties);
}

void _addOctreeFieldRenderer(brayns::Engine &engine)
{
    PLUGIN_INFO("Registering 'octree_field_renderer' renderer");
    brayns::PropertyMap properties;
    properties.setProperty({"exposure", 1., 1., 10., {"Exposure"}});
    properties.setProperty({"minRayStep", 0.00001, 0.00001, 1.0, {"Smallest ray step"}});
    properties.setProperty({"nbRaySteps", 8, 2, 2048, {"Number of ray marching steps"}});
    properties.setProperty({"nbRayRefinementSteps", 8, 1, 128, {"Number of ray marching refinement steps"}});
    properties.setProperty({"cutoff", 2000.0, 0.0, 1e5, {"cutoff"}});
    properties.setProperty({"alphaCorrection", 1.0, 0.001, 1.0, {"Alpha correction"}});
    engine.addRendererType("octree_field_renderer", properties);
}

FieldRendererPlugin::FieldRendererPlugin(const std::string &connectionString, const std::string &schema,
                                         const bool useOctree, const bool loadGeometry, const bool useCompartments)
    : ExtensionPlugin()
    , _uri(connectionString)
    , _schema(schema)
    , _useOctree(useOctree)
    , _loadGeometry(loadGeometry)
    , _useCompartments(useCompartments)
{
}

void FieldRendererPlugin::init()
{
    auto actionInterface = _api->getActionInterface();
    auto &engine = _api->getEngine();

    if (actionInterface)
    {
        std::string entryPoint = PLUGIN_API_PREFIX + "version";
        PLUGIN_INFO("Registering '" + entryPoint + "' endpoint");
        actionInterface->registerRequest<Response>(entryPoint, [&]() { return _version(); });

        entryPoint = PLUGIN_API_PREFIX + "set-simulation-parameters";
        PLUGIN_INFO("Registering '" + entryPoint + "' endpoint");
        actionInterface->registerRequest<SimulationParameters, Response>(entryPoint,
                                                                         [&](const SimulationParameters &payload) {
                                                                             return _setSimulationParameters(payload);
                                                                         });
    }

    if (_useOctree)
    {
        _addOctreeFieldRenderer(engine);
        _api->getParametersManager().getRenderingParameters().setCurrentRenderer("octree_field_renderer");
    }
    else
    {
        _addFieldRenderer(engine);
        _api->getParametersManager().getRenderingParameters().setCurrentRenderer("field_renderer");
    }
    _attachFieldsHandler();
}

Response FieldRendererPlugin::_version() const
{
    Response response;
    response.contents = PLUGIN_VERSION;
    return response;
}

void FieldRendererPlugin::_attachFieldsHandler()
{
    auto &scene = _api->getScene();
    auto model = scene.createModel();
    const auto &size = Vector3f(1.f);

    const size_t materialId = 0;
    auto material = model->createMaterial(materialId, "default");

    if (_useOctree)
    {
        OctreeFieldHandlerPtr handler = std::make_shared<OctreeFieldHandler>(_uri, _schema, _useCompartments);
        model->setSimulationHandler(handler);
    }
    else
    {
        FieldHandlerPtr handler = std::make_shared<FieldHandler>(_uri, _schema, _useCompartments);
        model->setSimulationHandler(handler);
    }
    setTransferFunction(model->getTransferFunction());

    DBConnector connector(_uri, _schema, _useCompartments);
    if (_loadGeometry)
    {
        const float radius = 2.f;
        for (auto point : connector.getAllNeurons())
            model->addSphere(materialId, {{point.x, point.y, point.z}, radius});
    }
    else
    {
        const auto box = connector.getCircuitBoundingBox();
        const auto halfSize = box.getSize() / 2.f;
        model->addSphere(materialId, {box.getCenter(), std::max(halfSize.x, std::max(halfSize.y, halfSize.z))});
    }

    auto modelDescriptor = std::make_shared<ModelDescriptor>(std::move(model), "Field gizmo");
    scene.addModel(modelDescriptor);
}

Response FieldRendererPlugin::_setSimulationParameters(const SimulationParameters &payload)
{
    Response response;
    try
    {
        if (payload.density > 100)
            PLUGIN_THROW(std::runtime_error("Density cannot be more than 100%"));

        auto &scene = _api->getScene();
        auto modelDescriptor = scene.getModelDescriptors()[0];
        if (modelDescriptor)
        {
            auto handler = modelDescriptor->getModel().getSimulationHandler();
            if (!handler)
                PLUGIN_THROW(std::runtime_error("Model has no simulation handler"));

            if (_useOctree)
            {
                OctreeFieldHandler *fieldHandler = dynamic_cast<OctreeFieldHandler *>(handler.get());
                if (!fieldHandler)
                    PLUGIN_THROW(
                        std::runtime_error("Model simulation handler is not an octree-based field "
                                           "simulation handler"));
                fieldHandler->setParams(payload);
            }
            else
            {
                FieldHandler *fieldHandler = dynamic_cast<FieldHandler *>(handler.get());
                if (!fieldHandler)
                    PLUGIN_THROW(
                        std::runtime_error("Model simulation handler is not a "
                                           "field simulation handler"));
                fieldHandler->setParams(payload);
            }
        }
        else
            PLUGIN_THROW(std::runtime_error("Model does not exist"));
    }
    CATCH_STD_EXCEPTION()
    return response;
}

extern "C" ExtensionPlugin *brayns_plugin_create(int argc, char **argv)
{
    PLUGIN_INFO("");
    PLUGIN_INFO("_|_|_|_|  _|            _|        _|            ");
    PLUGIN_INFO("_|              _|_|    _|    _|_|_|    _|_|_|  ");
    PLUGIN_INFO("_|_|_|    _|  _|_|_|_|  _|  _|    _|  _|_|      ");
    PLUGIN_INFO("_|        _|  _|        _|  _|    _|      _|_|  ");
    PLUGIN_INFO("_|        _|    _|_|_|  _|    _|_|_|  _|_|_|    ");
    PLUGIN_INFO("");
    PLUGIN_INFO("Initializing MEG plug-in (version " << PLUGIN_VERSION << ")");
    PLUGIN_INFO("");

    std::string uri;
    std::string schema;

    bool useOctree{false};
    bool loadGeometry{false};
    bool useCompartments{false};

    const std::string KEY_SCHEMA = "schema=";
    const std::string KEY_USE_OCTREE = "--use-octree";
    const std::string KEY_LOAD_GEOMETRY = "--load-geometry";
    const std::string KEY_USE_COMPARTMENTS = "--use-compartments";

    for (size_t i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument.find(KEY_SCHEMA) != -1)
            schema = argument.substr(KEY_SCHEMA.length());
        else if (argument.find(KEY_USE_OCTREE) != -1)
            useOctree = true;
        else if (argument.find(KEY_LOAD_GEOMETRY) != -1)
            loadGeometry = true;
        else if (argument.find(KEY_USE_COMPARTMENTS) != -1)
            useCompartments = true;
        else
            uri += argument + " ";
    }
    PLUGIN_INFO("URI: " << uri << ", Schema: " << schema);

    return new FieldRendererPlugin(uri, schema, useOctree, loadGeometry, useCompartments);
}

} // namespace fieldrenderer
