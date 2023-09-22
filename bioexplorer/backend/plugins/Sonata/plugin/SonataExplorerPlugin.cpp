/*
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include "SonataExplorerPlugin.h"

#include <Defines.h>
#include <Version.h>

#include <common/Logs.h>
#include <common/Properties.h>

#include <plugin/io/SonataCacheLoader.h>
#include <plugin/neuroscience/astrocyte/AstrocyteLoader.h>
#include <plugin/neuroscience/common/MorphologyLoader.h>
#include <plugin/neuroscience/neuron/AdvancedCircuitLoader.h>
#include <plugin/neuroscience/neuron/CellGrowthHandler.h>
#include <plugin/neuroscience/neuron/MEGHandler.h>
#include <plugin/neuroscience/neuron/MeshCircuitLoader.h>
#include <plugin/neuroscience/neuron/MorphologyCollageLoader.h>
#include <plugin/neuroscience/neuron/PairSynapsesLoader.h>
#include <plugin/neuroscience/neuron/SpikeSimulationHandler.h>
#include <plugin/neuroscience/neuron/SynapseCircuitLoader.h>
#include <plugin/neuroscience/neuron/VoltageSimulationHandler.h>

#include <plugin/meshing/PointCloudMesher.h>

#include <platform/core/common/ActionInterface.h>
#include <platform/core/common/Progress.h>
#include <platform/core/common/Properties.h>
#include <platform/core/common/Timer.h>
#include <platform/core/common/geometry/Streamline.h>
#include <platform/core/common/utils/ImageUtils.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>

#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <random>
#include <regex>
#include <unistd.h>

#include <sys/types.h>
#include <sys/wait.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Skin_surface_3.h>
#include <CGAL/Union_of_balls_3.h>
#include <CGAL/mesh_skin_surface_3.h>
#include <CGAL/mesh_union_of_balls_3.h>
#include <CGAL/subdivide_union_of_balls_mesh_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Skin_surface_traits_3<K> Traits;
typedef K::Point_3 Point_3;
typedef K::Weighted_point_3 Weighted_point;
typedef CGAL::Polyhedron_3<K> Polyhedron;
typedef CGAL::Skin_surface_traits_3<K> Traits;
typedef CGAL::Skin_surface_3<Traits> Skin_surface_3;
typedef CGAL::Union_of_balls_3<Traits> Union_of_balls_3;

#define CATCH_STD_EXCEPTION()           \
    catch (const std::runtime_error& e) \
    {                                   \
        response.status = false;        \
        response.contents = e.what();   \
        PLUGIN_ERROR(e.what());         \
    }

using namespace core;

namespace sonataexplorer
{
using namespace api;
using namespace io;
using namespace loader;
using namespace neuroscience;
using namespace neuron;
using namespace astrocyte;

#define REGISTER_LOADER(LOADER, FUNC) registry.registerLoader({std::bind(&LOADER::getSupportedDataTypes), FUNC});

const std::string PLUGIN_API_PREFIX = "se-";

void _addGrowthRenderer(Engine& engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_CELL_GROWTH);
    PropertyMap properties;
    properties.setProperty(RENDERER_PROPERTY_ALPHA_CORRECTION);
    properties.setProperty(SONATA_RENDERER_PROPERTY_CELL_GROWTH_SIMULATION_THRESHOLD);
    properties.setProperty(COMMON_PROPERTY_EXPOSURE);
    properties.setProperty(RENDERER_PROPERTY_FOG_START);
    properties.setProperty(RENDERER_PROPERTY_FOG_THICKNESS);
    properties.setProperty(SONATA_RENDERER_PROPERTY_USE_TRANSFER_FUNCTION_COLOR);
    properties.setProperty(RENDERER_PROPERTY_SHADOW_INTENSITY);
    properties.setProperty(RENDERER_PROPERTY_SOFT_SHADOW_STRENGTH);
    properties.setProperty(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_RAY_LENGTH);
    properties.setProperty(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER);
    engine.addRendererType(RENDERER_CELL_GROWTH, properties);
}

void _addProximityRenderer(Engine& engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_PROXIMITY);
    PropertyMap properties;
    properties.setProperty(RENDERER_PROPERTY_ALPHA_CORRECTION);
    properties.setProperty(SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_DISTANCE);
    properties.setProperty(SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_FAR_COLOR);
    properties.setProperty(SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_NEAR_COLOR);
    properties.setProperty(SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_DIFFERENT_MATERIAL);
    properties.setProperty(SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_SURFACE_SHADING_ENABLED);
    properties.setProperty(RENDERER_PROPERTY_MAX_RAY_DEPTH);
    properties.setProperty(COMMON_PROPERTY_EXPOSURE);
    properties.setProperty(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER);
    engine.addRendererType(RENDERER_PROXIMITY, properties);
}

void _addSphereClippingPerspectiveCamera(Engine& engine)
{
    PLUGIN_REGISTER_CAMERA(CAMERA_SPHERE_CLIPPING_PERSPECTIVE);
    PropertyMap properties;
    properties.setProperty(CAMERA_PROPERTY_FIELD_OF_VIEW);
    properties.setProperty(CAMERA_PROPERTY_ASPECT_RATIO);
    properties.setProperty(CAMERA_PROPERTY_APERTURE_RADIUS);
    properties.setProperty(CAMERA_PROPERTY_FOCAL_DISTANCE);
    properties.setProperty(CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES);
    properties.setProperty(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER);
    engine.addCameraType(CAMERA_SPHERE_CLIPPING_PERSPECTIVE, properties);
}

std::string _sanitizeString(const std::string& input)
{
    static const std::vector<std::string> sanitetizeItems = {"\"", "\\", "'", ";", "&", "|", "`"};

    std::string result = "";

    for (size_t i = 0; i < input.size(); i++)
    {
        bool found = false;
        for (const auto& token : sanitetizeItems)
        {
            if (std::string(1, input[i]) == token)
            {
                result += "\\" + token;
                found = true;
                break;
            }
        }
        if (!found)
        {
            result += std::string(1, input[i]);
        }
    }
    return result;
}

std::vector<std::string> _splitString(const std::string& source, const char token)
{
    std::vector<std::string> result;
    std::string split;
    std::istringstream ss(source);
    while (std::getline(ss, split, token))
        result.push_back(split);

    return result;
}

SonataExplorerPlugin::SonataExplorerPlugin()
    : ExtensionPlugin()
{
}

void SonataExplorerPlugin::init()
{
    auto& scene = _api->getScene();
    auto& registry = scene.getLoaderRegistry();
    auto& pm = _api->getParametersManager();

    // Loaders
    PLUGIN_REGISTER_LOADER(LOADER_BRICK);
    registry.registerLoader(std::make_unique<SonataCacheLoader>(scene, SonataCacheLoader::getCLIProperties()));

    PLUGIN_REGISTER_LOADER(LOADER_SYNAPSE_CIRCUIT);
    registry.registerLoader(std::make_unique<SynapseCircuitLoader>(scene, pm.getApplicationParameters(),
                                                                   SynapseCircuitLoader::getCLIProperties()));

    PLUGIN_REGISTER_LOADER(LOADER_MORPHOLOGY);
    registry.registerLoader(std::make_unique<neuroscience::common::MorphologyLoader>(
        scene, neuroscience::common::MorphologyLoader::getCLIProperties()));

    PLUGIN_REGISTER_LOADER(LOADER_ADVANCED_CIRCUIT);
    registry.registerLoader(std::make_unique<AdvancedCircuitLoader>(scene, pm.getApplicationParameters(),
                                                                    AdvancedCircuitLoader::getCLIProperties()));

    PLUGIN_REGISTER_LOADER(LOADER_MORPHOLOGY_COLLAGE);
    registry.registerLoader(std::make_unique<MorphologyCollageLoader>(scene, pm.getApplicationParameters(),
                                                                      MorphologyCollageLoader::getCLIProperties()));

    PLUGIN_REGISTER_LOADER(LOADER_MESH_CIRCUIT);
    registry.registerLoader(std::make_unique<MeshCircuitLoader>(scene, pm.getApplicationParameters(),
                                                                MeshCircuitLoader::getCLIProperties()));

    PLUGIN_REGISTER_LOADER(LOADER_PAIR_SYNAPSE);
    registry.registerLoader(std::make_unique<PairSynapsesLoader>(scene, pm.getApplicationParameters(),
                                                                 PairSynapsesLoader::getCLIProperties()));

    PLUGIN_REGISTER_LOADER(LOADER_ASTROCYTES);
    registry.registerLoader(
        std::make_unique<AstrocyteLoader>(scene, pm.getApplicationParameters(), AstrocyteLoader::getCLIProperties()));

    // Renderers
    auto& engine = _api->getEngine();
    auto& params = engine.getParametersManager().getApplicationParameters();
    const auto& engineName = params.getEngine();
    if (engineName == ENGINE_OSPRAY)
    {
        _addGrowthRenderer(engine);
        _addProximityRenderer(engine);
        _addSphereClippingPerspectiveCamera(engine);
    }

    // End-points
    auto actionInterface = _api->getActionInterface();
    if (actionInterface)
    {
        std::string endPoint = PLUGIN_API_PREFIX + "get-version";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<Response>(endPoint, [&]() { return _getVersion(); });

        endPoint = PLUGIN_API_PREFIX + "save-model-to-cache";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerNotification<ExportModelToFile>(endPoint, [&](const ExportModelToFile& param)
                                                                 { _exportModelToFile(param); });

        endPoint = PLUGIN_API_PREFIX + "save-model-to-mesh";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerNotification<ExportModelToMesh>(endPoint, [&](const ExportModelToMesh& param)
                                                                 { _exportModelToMesh(param); });

        endPoint = PLUGIN_API_PREFIX + "set-connections-per-value";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerNotification<ConnectionsPerValue>(endPoint, [&](const ConnectionsPerValue& param)
                                                                   { _setConnectionsPerValue(param); });

        endPoint = PLUGIN_API_PREFIX + "attach-cell-growth-handler";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerNotification<AttachCellGrowthHandler>(endPoint,
                                                                                  [&](const AttachCellGrowthHandler& s)
                                                                                  { _attachCellGrowthHandler(s); });

        endPoint = PLUGIN_API_PREFIX + "attach-circuit-simulation-handler";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerNotification<AttachCircuitSimulationHandler>(
            endPoint, [&](const AttachCircuitSimulationHandler& s) { _attachCircuitSimulationHandler(s); });

        endPoint = PLUGIN_API_PREFIX + "set-spike-report-visualization-settings";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerNotification<SpikeReportVisualizationSettings>(
            endPoint, [&](const SpikeReportVisualizationSettings& s) { _setSpikeReportVisualizationSettings(s); });

        endPoint = PLUGIN_API_PREFIX + "add-column";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerNotification<AddColumn>(endPoint, [&](const AddColumn& details)
                                                                    { _addColumn(details); });

        endPoint = PLUGIN_API_PREFIX + "add-sphere";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<AddSphere, Response>(endPoint, [&](const AddSphere& details)
                                                                         { return _addSphere(details); });

        endPoint = PLUGIN_API_PREFIX + "add-pill";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<AddPill, Response>(endPoint, [&](const AddPill& details)
                                                                       { return _addPill(details); });

        endPoint = PLUGIN_API_PREFIX + "add-cylinder";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<AddCylinder, Response>(endPoint, [&](const AddCylinder& details)
                                                                           { return _addCylinder(details); });

        endPoint = PLUGIN_API_PREFIX + "add-box";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<AddBox, Response>(endPoint, [&](const AddBox& details)
                                                                      { return _addBox(details); });

        endPoint = PLUGIN_API_PREFIX + "load-meg";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<LoadMEGSettings, Response>(endPoint,
                                                                               [&](const LoadMEGSettings& details)
                                                                               { return _loadMEG(details); });
    }
}

void SonataExplorerPlugin::preRender()
{
    if (_dirty)
    {
        auto& scene = _api->getScene();
        auto& engine = _api->getEngine();
        scene.markModified();
        engine.triggerRender();
    }
    _dirty = false;
}

Response SonataExplorerPlugin::_getVersion() const
{
    Response response;
    response.contents = PACKAGE_VERSION_STRING;
    return response;
}

Response SonataExplorerPlugin::_exportModelToFile(const ExportModelToFile& saveModel)
{
    Response response;
    try
    {
        auto& scene = _api->getScene();
        auto modelDescriptor = scene.getModel(saveModel.modelId);
        if (modelDescriptor)
        {
            SonataCacheLoader brickLoader(_api->getScene());
            brickLoader.exportToFile(modelDescriptor, saveModel.path);
        }
        else
            PLUGIN_ERROR("Model " << saveModel.modelId << " is not registered");
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response SonataExplorerPlugin::_exportModelToMesh(const ExportModelToMesh& details)
{
    Response response;
    try
    {
        auto& scene = _api->getScene();
        auto modelDescriptor = scene.getModel(details.modelId);
        if (modelDescriptor)
        {
            const auto& model = modelDescriptor->getModel();
            std::list<Weighted_point> l;
            for (const auto& spheres : model.getSpheres())
            {
                uint64_t count = 0;
                for (const auto& s : spheres.second)
                {
                    if (count % details.density == 0)
                        l.push_front(Weighted_point(Point_3(s.center.x, s.center.y, s.center.z),
                                                    details.radiusMultiplier * s.radius));
                    ++count;
                }
            }

            PLUGIN_INFO("Constructing skin surface from " << l.size() << " spheres");

            Polyhedron polyhedron;
            if (details.skin)
            {
                Skin_surface_3 skinSurface(l.begin(), l.end(), details.shrinkFactor);

                PLUGIN_INFO("Meshing skin surface...");
                CGAL::mesh_skin_surface_3(skinSurface, polyhedron);
                CGAL::Polygon_mesh_processing::triangulate_faces(polyhedron);
            }
            else
            {
                Union_of_balls_3 union_of_balls(l.begin(), l.end());
                CGAL::mesh_union_of_balls_3(union_of_balls, polyhedron);
            }

            PLUGIN_INFO("Export mesh to " << details.path);
            std::ofstream out(details.path);
            out << polyhedron;
        }
        else
            PLUGIN_THROW("Model " + std::to_string(details.modelId) + " is not registered");
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response SonataExplorerPlugin::_setConnectionsPerValue(const ConnectionsPerValue& cpv)
{
    Response response;
    try
    {
        meshing::PointCloud pointCloud;

        auto& scene = _api->getScene();
        auto modelDescriptor = scene.getModel(cpv.modelId);
        if (modelDescriptor)
        {
            auto simulationHandler = modelDescriptor->getModel().getSimulationHandler();
            if (!simulationHandler)
                PLUGIN_THROW("Scene has not user data handler");

            auto& model = modelDescriptor->getModel();
            for (const auto& spheres : model.getSpheres())
            {
                for (const auto& s : spheres.second)
                {
                    const float* data = static_cast<float*>(simulationHandler->getFrameData(cpv.frame));

                    const float value = data[s.userData];
                    if (abs(value - cpv.value) < cpv.epsilon)
                        pointCloud[spheres.first].push_back({s.center.x, s.center.y, s.center.z, s.radius});
                }
            }

            if (!pointCloud.empty())
            {
                auto meshModel = scene.createModel();
                meshing::PointCloudMesher mesher;
                if (mesher.toConvexHull(*meshModel, pointCloud))
                {
                    auto modelDesc =
                        std::make_shared<ModelDescriptor>(std::move(meshModel),
                                                          "Connection for value " + std::to_string(cpv.value));
                    scene.addModel(modelDesc);
                    _markModified();
                }
            }
            else
                PLUGIN_INFO("No connections added for value " << std::to_string(cpv.value));
        }
        else
            PLUGIN_INFO("Model " << cpv.modelId << " is not registered");
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response SonataExplorerPlugin::_attachCellGrowthHandler(const AttachCellGrowthHandler& details)
{
    Response response;
    try
    {
        PLUGIN_INFO("Attaching Cell Growth Handler to model " << details.modelId);
        auto& scene = _api->getScene();
        auto modelDescriptor = scene.getModel(details.modelId);
        if (modelDescriptor)
        {
            auto handler = std::make_shared<CellGrowthHandler>(details.nbFrames);
            modelDescriptor->getModel().setSimulationHandler(handler);
        }
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response SonataExplorerPlugin::_attachCircuitSimulationHandler(const AttachCircuitSimulationHandler& details)
{
    Response response;
    try
    {
        PLUGIN_INFO("Attaching Circuit Simulation Handler to model " << details.modelId);
        auto& scene = _api->getScene();
        auto modelDescriptor = scene.getModel(details.modelId);
        if (modelDescriptor)
        {
            const brion::BlueConfig blueConfiguration(details.circuitConfiguration);
            const brain::Circuit circuit(blueConfiguration);
            auto gids = circuit.getGIDs();
            auto handler = std::make_shared<VoltageSimulationHandler>(
                blueConfiguration.getReportSource(details.reportName).getPath(), gids, details.synchronousMode);
            auto& model = modelDescriptor->getModel();
            model.setSimulationHandler(handler);
            AdvancedCircuitLoader::setSimulationTransferFunction(model.getTransferFunction());
        }
        else
            PLUGIN_THROW("Model " + std::to_string(details.modelId) + " does not exist");
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response SonataExplorerPlugin::_setSpikeReportVisualizationSettings(const SpikeReportVisualizationSettings& payload)
{
    Response response;
    try
    {
        PLUGIN_INFO("Setting spike report visualization settings to model " << payload.modelId);
        auto& scene = _api->getScene();
        auto modelDescriptor = scene.getModel(payload.modelId);
        if (!modelDescriptor)
            PLUGIN_THROW("Invalid model id");
        auto handler = modelDescriptor->getModel().getSimulationHandler();
        if (!handler)
            PLUGIN_THROW("Model has no simulation handler");
        auto spikeHandler = dynamic_cast<SpikeSimulationHandler*>(handler.get());
        if (!spikeHandler)
            PLUGIN_THROW("Model does not hold a spike report simulation handler");
        spikeHandler->setVisualizationSettings(payload.restVoltage, payload.spikingVoltage, payload.timeInterval,
                                               payload.decaySpeed);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response SonataExplorerPlugin::_loadMEG(const LoadMEGSettings& details)
{
    Response response;
    try
    {
        PLUGIN_INFO("Loading MEG for circuit " << details.path);
        auto& scene = _api->getScene();
        auto handler = std::make_shared<MEGHandler>(details.path, details.reportName, details.synchronous);
        if (!handler)
            PLUGIN_THROW("Failed to handler");
        auto model = scene.createModel();
        if (!model)
            PLUGIN_THROW("Failed to create model");
        const auto metadata = handler->buildModel(*model, details.voxelSize, details.density);
        model->setSimulationHandler(handler);
        scene.addModel(std::make_shared<ModelDescriptor>(std::move(model), details.name, metadata));
        scene.markModified();
        _markModified();
    }
    CATCH_STD_EXCEPTION()
    return response;
}

void SonataExplorerPlugin::_createShapeMaterial(ModelPtr& model, const size_t id, const Vector3d& color,
                                                const double& opacity)
{
    MaterialPtr material = model->createMaterial(id, std::to_string(id));
    material->setDiffuseColor(color);
    material->setOpacity(opacity);
    material->setSpecularExponent(0.0);
    material->setShadingMode(MaterialShadingMode::diffuse_transparency);
    material->markModified();
}

Response SonataExplorerPlugin::_addSphere(const AddSphere& details)
{
    Response response;
    try
    {
        if (details.center.size() < 3)
            PLUGIN_THROW(
                "Sphere center has the wrong number of parameters (3 "
                "necessary)");
        if (details.color.size() < 4)
            PLUGIN_THROW(
                "Sphere color has the wrong number of parameters (RGBA, 4 "
                "necessary)");
        if (details.radius < 0.f)
            PLUGIN_THROW("Negative radius passed for sphere creation");

        auto& scene = _api->getScene();
        ModelPtr modelptr = scene.createModel();

        const size_t matId = 1;
        const Vector3d color(details.color[0], details.color[1], details.color[2]);
        const double opacity = details.color[3];
        _createShapeMaterial(modelptr, matId, color, opacity);

        const Vector3f center(details.center[0], details.center[1], details.center[2]);
        modelptr->addSphere(matId, {center, details.radius});

        size_t numModels = scene.getNumModels();
        const std::string name = details.name.empty() ? "sphere_" + std::to_string(numModels) : details.name;
        scene.addModel(std::make_shared<ModelDescriptor>(std::move(modelptr), name));
        scene.markModified();
        _markModified();
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response SonataExplorerPlugin::_addPill(const AddPill& details)
{
    Response response;
    try
    {
        if (details.p1.size() < 3)
            PLUGIN_THROW(
                "Pill point 1 has the wrong number of parameters (3 "
                "necessary)");
        if (details.p2.size() < 3)
            PLUGIN_THROW(
                "Pill point 2 has the wrong number of parameters (3 "
                "necessary)");
        if (details.color.size() < 4)
            PLUGIN_THROW(
                "Pill color has the wrong number of parameters (RGBA, 4 "
                "necessary)");
        if (details.type != "pill" && details.type != "conepill" && details.type != "sigmoidpill")
            PLUGIN_THROW(
                "Unknown pill type parameter. Must be either \"pill\", "
                "\"conepill\", or \"sigmoidpill\"");
        if (details.radius1 < 0.f || details.radius2 < 0.f)
            PLUGIN_THROW("Negative radius passed for the pill creation");

        auto& scene = _api->getScene();
        auto modelptr = scene.createModel();

        size_t matId = 1;
        const Vector3d color(details.color[0], details.color[1], details.color[2]);
        const double opacity = details.color[3];
        _createShapeMaterial(modelptr, matId, color, opacity);

        const Vector3f p0(details.p1[0], details.p1[1], details.p1[2]);
        const Vector3f p1(details.p2[0], details.p2[1], details.p2[2]);
        SDFGeometry sdf;
        if (details.type == "pill")
        {
            sdf = createSDFPill(p0, p1, details.radius1);
        }
        else if (details.type == "conepill")
        {
            sdf = createSDFConePill(p0, p1, details.radius1, details.radius2);
        }
        else if (details.type == "sigmoidpill")
        {
            sdf = createSDFConePillSigmoid(p0, p1, details.radius1, details.radius2);
        }

        modelptr->addSDFGeometry(matId, sdf, {});
        size_t numModels = scene.getNumModels();
        const std::string name = details.name.empty() ? details.type + "_" + std::to_string(numModels) : details.name;
        scene.addModel(std::make_shared<ModelDescriptor>(std::move(modelptr), name));
        _markModified();
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response SonataExplorerPlugin::_addCylinder(const AddCylinder& details)
{
    Response response;
    try
    {
        if (details.center.size() < 3)
            PLUGIN_THROW(
                "Cylinder center has the wrong number of parameters (3 "
                "necessary)");
        if (details.up.size() < 3)
            PLUGIN_THROW(
                "Cylinder up has the wrong number of parameters (3 "
                "necessary)");
        if (details.color.size() < 4)
            PLUGIN_THROW(
                "Cylinder color has the wrong number of parameters (RGBA, "
                "4 "
                "necessary)");
        if (details.radius < 0.0f)
            PLUGIN_THROW("Negative radius passed for cylinder creation");

        auto& scene = _api->getScene();
        ModelPtr modelptr = scene.createModel();

        const size_t matId = 1;
        const Vector3d color(details.color[0], details.color[1], details.color[2]);
        const double opacity = details.color[3];
        _createShapeMaterial(modelptr, matId, color, opacity);

        const Vector3f center(details.center[0], details.center[1], details.center[2]);
        const Vector3f up(details.up[0], details.up[1], details.up[2]);
        modelptr->addCylinder(matId, {center, up, details.radius});

        size_t numModels = scene.getNumModels();
        const std::string name = details.name.empty() ? "cylinder_" + std::to_string(numModels) : details.name;
        scene.addModel(std::make_shared<ModelDescriptor>(std::move(modelptr), name));
        _markModified();
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response SonataExplorerPlugin::_addBox(const AddBox& details)
{
    Response response;
    try
    {
        if (details.minCorner.size() < 3)
            PLUGIN_THROW(
                "Box minCorner has the wrong number of parameters (3 "
                "necessary)");
        if (details.maxCorner.size() < 3)
            PLUGIN_THROW(
                "Box maxCorner has the wrong number of parameters (3 "
                "necesary)");
        if (details.color.size() < 4)
            PLUGIN_THROW(
                "Box color has the wrong number of parameters (RGBA, 4 "
                "necesary)");

        auto& scene = _api->getScene();
        auto modelptr = scene.createModel();

        const size_t matId = 1;
        const Vector3d color(details.color[0], details.color[1], details.color[2]);
        const double opacity = details.color[3];
        _createShapeMaterial(modelptr, matId, color, opacity);

        const Vector3f minCorner(details.minCorner[0], details.minCorner[1], details.minCorner[2]);
        const Vector3f maxCorner(details.maxCorner[0], details.maxCorner[1], details.maxCorner[2]);

        TriangleMesh mesh = createBox(minCorner, maxCorner);

        modelptr->getTriangleMeshes()[matId] = mesh;
        modelptr->markInstancesDirty();

        size_t numModels = scene.getNumModels();
        const std::string name = details.name.empty() ? "box_" + std::to_string(numModels) : details.name;
        scene.addModel(std::make_shared<ModelDescriptor>(std::move(modelptr), name));
        _markModified();
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response SonataExplorerPlugin::_addColumn(const AddColumn& details)
{
    Response response;
    try
    {
        PLUGIN_INFO("Building Column model");

        auto& scene = _api->getScene();
        auto model = scene.createModel();

        const Vector3f white = {1.f, 1.f, 1.F};

        auto material = model->createMaterial(0, "column");
        material->setDiffuseColor(white);

        const Vector3fs verticesBottom = {{-0.25f, -1.0f, -0.5f}, {0.25f, -1.0f, -0.5f}, {0.5f, -1.0f, -0.25f},
                                          {0.5f, -1.0f, 0.25f},   {0.5f, -1.0f, -0.25f}, {0.5f, -1.0f, 0.25f},
                                          {0.25f, -1.0f, 0.5f},   {-0.25f, -1.0f, 0.5f}, {-0.5f, -1.0f, 0.25f},
                                          {-0.5f, -1.0f, -0.25f}};
        const Vector3fs verticesTop = {{-0.25f, 1.f, -0.5f}, {0.25f, 1.f, -0.5f}, {0.5f, 1.f, -0.25f},
                                       {0.5f, 1.f, 0.25f},   {0.5f, 1.f, -0.25f}, {0.5f, 1.f, 0.25f},
                                       {0.25f, 1.f, 0.5f},   {-0.25f, 1.f, 0.5f}, {-0.5f, 1.f, 0.25f},
                                       {-0.5f, 1.f, -0.25f}};

        const auto r = details.radius;
        for (size_t i = 0; i < verticesBottom.size(); ++i)
        {
            model->addCylinder(0, {verticesBottom[i], verticesBottom[(i + 1) % verticesBottom.size()], r / 2.f});
            model->addSphere(0, {verticesBottom[i], r});
        }

        for (size_t i = 0; i < verticesTop.size(); ++i)
        {
            model->addCylinder(0, {verticesTop[i], verticesTop[(i + 1) % verticesTop.size()], r / 2.f});
            model->addSphere(0, {verticesTop[i], r});
        }

        for (size_t i = 0; i < verticesTop.size(); ++i)
            model->addCylinder(0, {verticesBottom[i], verticesTop[i], r / 2.f});

        scene.addModel(std::make_shared<ModelDescriptor>(std::move(model), "Column"));
    }
    CATCH_STD_EXCEPTION()
    return response;
}

extern "C" ExtensionPlugin* core_plugin_create(int /*argc*/, char** /*argv*/)
{
    PLUGIN_INFO("");

    PLUGIN_INFO(
        "   _|_|_|                                  _|                _|_|_|_|                      _|                 "
        "                         ");
    PLUGIN_INFO(
        " _|          _|_|    _|_|_|      _|_|_|  _|_|_|_|    _|_|_|  _|        _|    _|  _|_|_|    _|    _|_|    _|  "
        "_|_|    _|_|    _|  _|_|  ");
    PLUGIN_INFO(
        "   _|_|    _|    _|  _|    _|  _|    _|    _|      _|    _|  _|_|_|      _|_|    _|    _|  _|  _|    _|  _|_| "
        "     _|_|_|_|  _|_|      ");
    PLUGIN_INFO(
        "       _|  _|    _|  _|    _|  _|    _|    _|      _|    _|  _|        _|    _|  _|    _|  _|  _|    _|  _|   "
        "     _|        _|        ");
    PLUGIN_INFO(
        " _|_|_|      _|_|    _|    _|    _|_|_|      _|_|    _|_|_|  _|_|_|_|  _|    _|  _|_|_|    _|    _|_|    _|   "
        "       _|_|_|  _|        ");
    PLUGIN_INFO(
        "                                                                                 _|                           "
        "                         ");
    PLUGIN_INFO(
        "                                                                                 _|                           "
        "                         ");
    PLUGIN_INFO("");
    PLUGIN_INFO("Initializing SonataExplorer plug-in");
    return new SonataExplorerPlugin();
}
} // namespace sonataexplorer
