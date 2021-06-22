/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#include "BioExplorerPlugin.h"

#include <plugin/biology/Assembly.h>
#include <plugin/common/CommonTypes.h>
#include <plugin/common/GeneralSettings.h>
#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>
#include <plugin/io/CacheLoader.h>
#include <plugin/io/OOCManager.h>

#include <brayns/common/ActionInterface.h>
#include <brayns/common/scene/ClipPlane.h>
#include <brayns/engineapi/Camera.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>
#include <brayns/parameters/ParametersManager.h>
#include <brayns/pluginapi/Plugin.h>

namespace bioexplorer
{
using namespace common;
using namespace io;

const std::string PLUGIN_API_PREFIX = "be-";

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)

#define CATCH_STD_EXCEPTION()           \
    catch (const std::runtime_error &e) \
    {                                   \
        response.status = false;        \
        response.contents = e.what();   \
        PLUGIN_ERROR(e.what());         \
    }

#define ASSEMBLY_CALL(__name__, __stmt__)               \
    Response response;                                  \
    try                                                 \
    {                                                   \
        auto it = _assemblies.find(__name__);           \
        if (it != _assemblies.end())                    \
            response.contents = (*it).second->__stmt__; \
        else                                            \
        {                                               \
            std::stringstream msg;                      \
            msg << "Assembly not found: " << __name__;  \
            PLUGIN_ERROR(msg.str());                    \
            response.status = false;                    \
            response.contents = msg.str();              \
        }                                               \
    }                                                   \
    CATCH_STD_EXCEPTION()                               \
    return response;

#define ASSEMBLY_CALL_VOID(__name__, __stmt__)         \
    Response response;                                 \
    try                                                \
    {                                                  \
        auto it = _assemblies.find(__name__);          \
        if (it != _assemblies.end())                   \
            (*it).second->__stmt__;                    \
        else                                           \
        {                                              \
            std::stringstream msg;                     \
            msg << "Assembly not found: " << __name__; \
            PLUGIN_ERROR(msg.str());                   \
            response.status = false;                   \
            response.contents = msg.str();             \
        }                                              \
    }                                                  \
    CATCH_STD_EXCEPTION()                              \
    return response;
#endif

Boxd vector_to_bounds(const std::vector<float> &lowBounds,
                      const std::vector<float> &highBounds)
{
    if (!lowBounds.empty() && lowBounds.size() != 3)
        PLUGIN_THROW("Invalid low bounds. 3 floats expected");
    if (!highBounds.empty() && highBounds.size() != 3)
        PLUGIN_THROW("Invalid high bounds. 3 floats expected");

    Boxd bounds;
    if (!lowBounds.empty())
        bounds.merge({lowBounds[0], lowBounds[1], lowBounds[2]});
    if (!highBounds.empty())
        bounds.merge({highBounds[0], highBounds[1], highBounds[2]});
    return bounds;
}

void _addBioExplorerRenderer(Engine &engine)
{
    PLUGIN_INFO("Registering 'bio_explorer' renderer");
    PropertyMap properties;
    properties.setProperty(
        {"giDistance", 10000., {"Global illumination distance"}});
    properties.setProperty(
        {"giWeight", 0., 1., 1., {"Global illumination weight"}});
    properties.setProperty(
        {"giSamples", 0, 0, 64, {"Global illumination samples"}});
    properties.setProperty({"shadows", 0., 0., 1., {"Shadow intensity"}});
    properties.setProperty({"softShadows", 0., 0., 1., {"Shadow softness"}});
    properties.setProperty(
        {"softShadowsSamples", 1, 1, 64, {"Soft shadow samples"}});
    properties.setProperty({"exposure", 1., 0.01, 10., {"Exposure"}});
    properties.setProperty({"fogStart", 0., 0., 1e6, {"Fog start"}});
    properties.setProperty({"fogThickness", 1e6, 1e6, 1e6, {"Fog thickness"}});
    properties.setProperty(
        {"maxBounces", 3, 1, 100, {"Maximum number of ray bounces"}});
    properties.setProperty({"useHardwareRandomizer",
                            false,
                            {"Use hardware accelerated randomizer"}});
    properties.setProperty({"showBackground", false, {"Show background"}});
    engine.addRendererType("bio_explorer", properties);
}

void _addBioExplorerFieldsRenderer(Engine &engine)
{
    PLUGIN_INFO("Registering 'bio_explorer_fields' renderer");
    PropertyMap properties;
    properties.setProperty({"exposure", 1., 1., 10., {"Exposure"}});
    properties.setProperty({"useHardwareRandomizer",
                            false,
                            {"Use hardware accelerated randomizer"}});
    properties.setProperty(
        {"minRayStep", 0.00001, 0.00001, 1.0, {"Smallest ray step"}});
    properties.setProperty(
        {"nbRaySteps", 8, 1, 2048, {"Number of ray marching steps"}});
    properties.setProperty({"nbRayRefinementSteps",
                            8,
                            1,
                            1000,
                            {"Number of ray marching refinement steps"}});
    properties.setProperty({"cutoff", 2000.0, 0.0, 1e5, {"cutoff"}});
    properties.setProperty(
        {"alphaCorrection", 1.0, 0.001, 1.0, {"Alpha correction"}});
    engine.addRendererType("bio_explorer_fields", properties);
}

void _addBioExplorerDensityRenderer(Engine &engine)
{
    PLUGIN_INFO("Registering 'bio_explorer_density' renderer");
    PropertyMap properties;
    properties.setProperty({"exposure", 1.5, 1., 10., {"Exposure"}});
    properties.setProperty({"rayStep", 2.0, 1.0, 1024.0, {"Ray marchingstep"}});
    properties.setProperty(
        {"sampleCount", 4, 1, 2048, {"Number of ray marching samples"}});
    properties.setProperty(
        {"searchLength", 1.0, 0.0001, 4096.0, {"Sample search length"}});
    properties.setProperty({"farPlane", 1000.0, 1.0, 1e6, {"Far plane"}});
    properties.setProperty(
        {"alphaCorrection", 1.0, 0.001, 1.0, {"Alpha correction"}});
    engine.addRendererType("bio_explorer_density", properties);
}

void _addBioExplorerPathTracingRenderer(Engine &engine)
{
    PLUGIN_INFO("Registering 'bio_explorer_path_tracing' renderer");
    PropertyMap properties;
    properties.setProperty({"exposure", 1., 0.1, 10., {"Exposure"}});
    properties.setProperty({"useHardwareRandomizer",
                            false,
                            {"Use hardware accelerated randomizer"}});
    properties.setProperty({"showBackground", false, {"Show background"}});
    properties.setProperty(
        {"aoStrength", 1.0, 0.0001, 10.0, {"Sample search strength"}});
    properties.setProperty(
        {"aoDistance", 1e6, 0.1, 1e6, {"Sample search distance"}});
    engine.addRendererType("bio_explorer_path_tracing", properties);
}

void _addBioExplorerPerspectiveCamera(Engine &engine)
{
    PLUGIN_INFO("Registering 'bio_explorer_perspective' camera");

    PropertyMap properties;
    properties.setProperty({"fovy", 45., .1, 360., {"Field of view"}});
    properties.setProperty({"aspect", 1., {"Aspect ratio"}});
    properties.setProperty({"apertureRadius", 0., {"Aperture radius"}});
    properties.setProperty({"focusDistance", 1., {"Focus Distance"}});
    properties.setProperty({"enableClippingPlanes", true, {"Clipping"}});
    properties.setProperty({"stereo", false, {"Stereo"}});
    properties.setProperty(
        {"interpupillaryDistance", 0.0635, 0.0, 10.0, {"Eye separation"}});
    engine.addCameraType("bio_explorer_perspective", properties);
}

BioExplorerPlugin::BioExplorerPlugin(int argc, char **argv)
    : ExtensionPlugin()
{
    _parseCommandLineArguments(argc, argv);
}

void BioExplorerPlugin::init()
{
    auto actionInterface = _api->getActionInterface();
    auto &scene = _api->getScene();
    auto &camera = _api->getCamera();
    auto &engine = _api->getEngine();
    auto &registry = scene.getLoaderRegistry();

    registry.registerLoader(
        std::make_unique<CacheLoader>(scene, CacheLoader::getCLIProperties()));

    if (actionInterface)
    {
        std::string endPoint = PLUGIN_API_PREFIX + "get-version";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<Response>(endPoint, [&]()
                                                   { return _getVersion(); });

        endPoint = PLUGIN_API_PREFIX + "get-scene-information";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<SceneInformationDetails>(
            endPoint, [&]() { return _getSceneInformation(); });

        endPoint = PLUGIN_API_PREFIX + "set-general-settings";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<GeneralSettingsDetails, Response>(
            endPoint, [&](const GeneralSettingsDetails &payload)
            { return _setGeneralSettings(payload); });

        endPoint = PLUGIN_API_PREFIX + "reset";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<Response>(endPoint,
                                                   [&]() { return _reset(); });

        endPoint = PLUGIN_API_PREFIX + "remove-assembly";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<AssemblyDetails, Response>(
            endPoint, [&](const AssemblyDetails &payload)
            { return _removeAssembly(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-assembly";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<AssemblyDetails, Response>(
            endPoint, [&](const AssemblyDetails &payload)
            { return _addAssembly(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-protein-color-scheme";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<ColorSchemeDetails, Response>(
            endPoint, [&](const ColorSchemeDetails &payload)
            { return _setColorScheme(payload); });

        endPoint =
            PLUGIN_API_PREFIX + "set-protein-amino-acid-sequence-as-string";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface
            ->registerRequest<AminoAcidSequenceAsStringDetails, Response>(
                endPoint, [&](const AminoAcidSequenceAsStringDetails &payload)
                { return _setAminoAcidSequenceAsString(payload); });

        endPoint =
            PLUGIN_API_PREFIX + "set-protein-amino-acid-sequence-as-ranges";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface
            ->registerRequest<AminoAcidSequenceAsRangesDetails, Response>(
                endPoint, [&](const AminoAcidSequenceAsRangesDetails &payload)
                { return _setAminoAcidSequenceAsRanges(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-protein-amino-acid-information";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<AminoAcidInformationDetails, Response>(
            endPoint, [&](const AminoAcidInformationDetails &payload)
            { return _getAminoAcidInformation(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-protein-amino-acid";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<AminoAcidDetails, Response>(
            endPoint, [&](const AminoAcidDetails &payload)
            { return _setAminoAcid(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-protein-instance-transformation";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface
            ->registerRequest<ProteinInstanceTransformationDetails, Response>(
                endPoint,
                [&](const ProteinInstanceTransformationDetails &payload)
                { return _setProteinInstanceTransformation(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-protein-instance-transformation";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface
            ->registerRequest<ProteinInstanceTransformationDetails, Response>(
                endPoint,
                [&](const ProteinInstanceTransformationDetails &payload)
                { return _getProteinInstanceTransformation(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-rna-sequence";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<RNASequenceDetails, Response>(
            endPoint, [&](const RNASequenceDetails &payload)
            { return _addRNASequence(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-membrane";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<ParametricMembraneDetails, Response>(
            endPoint, [&](const ParametricMembraneDetails &payload)
            { return _addParametricMembrane(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-mesh-based-membrane";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<MeshBasedMembraneDetails, Response>(
            endPoint, [&](const MeshBasedMembraneDetails &payload)
            { return _addMeshBasedMembrane(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-protein";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<ProteinDetails, Response>(
            endPoint, [&](const ProteinDetails &payload)
            { return _addProtein(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-glycans";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<SugarsDetails, Response>(
            endPoint,
            [&](const SugarsDetails &payload) { return _addGlycans(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-sugars";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<SugarsDetails, Response>(
            endPoint,
            [&](const SugarsDetails &payload) { return _addSugars(payload); });

        endPoint = PLUGIN_API_PREFIX + "export-to-file";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<FileAccessDetails, Response>(
            endPoint, [&](const FileAccessDetails &payload)
            { return _exportToFile(payload); });

        endPoint = PLUGIN_API_PREFIX + "import-from-file";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<FileAccessDetails, Response>(
            endPoint, [&](const FileAccessDetails &payload)
            { return _importFromFile(payload); });

        endPoint = PLUGIN_API_PREFIX + "export-to-xyz";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<FileAccessDetails, Response>(
            endPoint, [&](const FileAccessDetails &payload)
            { return _exportToXYZ(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-grid";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerNotification<AddGridDetails>(
            endPoint,
            [&](const AddGridDetails &payload) { _addGrid(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-sphere";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerNotification<AddSphereDetails>(
            endPoint,
            [&](const AddSphereDetails &payload) { _addSphere(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-model-ids";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<IdsDetails>(endPoint,
                                                     [&]() -> IdsDetails {
                                                         return _getModelIds();
                                                     });

        endPoint = PLUGIN_API_PREFIX + "get-model-name";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<ModelIdDetails, ModelNameDetails>(
            endPoint,
            [&](const ModelIdDetails &payload) -> ModelNameDetails
            { return _getModelName(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-materials";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerNotification<MaterialsDetails>(
            endPoint,
            [&](const MaterialsDetails &payload) { _setMaterials(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-material-ids";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<ModelIdDetails, IdsDetails>(
            endPoint,
            [&](const ModelIdDetails &payload) -> IdsDetails
            { return _getMaterialIds(payload); });

        endPoint = PLUGIN_API_PREFIX + "build-fields";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<BuildFieldsDetails, Response>(
            endPoint, [&](const BuildFieldsDetails &payload)
            { return _buildFields(payload); });

        endPoint = PLUGIN_API_PREFIX + "export-fields-to-file";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<ModelIdFileAccessDetails, Response>(
            endPoint, [&](const ModelIdFileAccessDetails &payload)
            { return _exportFieldsToFile(payload); });

        endPoint = PLUGIN_API_PREFIX + "import-fields-from-file";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<FileAccessDetails, Response>(
            endPoint, [&](const FileAccessDetails &payload)
            { return _importFieldsFromFile(payload); });

        endPoint = PLUGIN_API_PREFIX + "build-point-cloud";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<BuildPointCloudDetails, Response>(
            endPoint, [&](const BuildPointCloudDetails &payload)
            { return _buildPointCloud(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-models-visibility";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<ModelsVisibilityDetails, Response>(
            endPoint, [&](const ModelsVisibilityDetails &payload)
            { return _setModelsVisibility(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-out-of-core-configuration";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<Response>(
            endPoint, [&]() { return _getOOCConfiguration(); });

        endPoint = PLUGIN_API_PREFIX + "get-out-of-core-progress";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<Response>(endPoint,
                                                   [&]() {
                                                       return _getOOCProgress();
                                                   });

        endPoint = PLUGIN_API_PREFIX + "get-out-of-core-average-loading-time";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<Response>(
            endPoint, [&]() { return _getOOCAverageLoadingTime(); });

#ifdef USE_PQXX
        endPoint = PLUGIN_API_PREFIX + "export-to-database";
        PLUGIN_INFO("Registering '" + endPoint + "' endpoint");
        actionInterface->registerRequest<DatabaseAccessDetails, Response>(
            endPoint, [&](const DatabaseAccessDetails &payload)
            { return _exportToDatabase(payload); });
#endif
    }

    // Module components
    _addBioExplorerPerspectiveCamera(engine);
    _addBioExplorerRenderer(engine);
    _addBioExplorerFieldsRenderer(engine);
    _addBioExplorerDensityRenderer(engine);
    _addBioExplorerPathTracingRenderer(engine);

    // Out-of-core
    if (_commandLineArguments.find(ARG_OOC_ENABLED) !=
        _commandLineArguments.end())
    {
        _oocManager =
            OOCManagerPtr(new OOCManager(scene, camera, _commandLineArguments));
        if (_oocManager->getShowGrid())
        {
            AddGridDetails grid;
            const auto &sceneConfiguration =
                _oocManager->getSceneConfiguration();
            const auto sceneSize = sceneConfiguration.sceneSize.x;
            const auto brickSize = sceneConfiguration.brickSize.x;
            grid.position = {-brickSize / 2.f, -brickSize / 2.f,
                             -brickSize / 2.f};
            grid.minValue = -sceneSize / 2.0;
            grid.maxValue = sceneSize / 2.0;
            grid.steps = brickSize;
            grid.showAxis = false;
            grid.showPlanes = false;
            grid.showFullGrid = true;
            grid.radius = 0.1f;
            _addGrid(grid);
        }
    }
}

void BioExplorerPlugin::_parseCommandLineArguments(int argc, char **argv)
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

void BioExplorerPlugin::preRender()
{
    if (_oocManager)
        if (!_oocManager->getFrameBuffer())
        {
            PLUGIN_INFO("Starting Out-Of-Core manager");
            auto &frameBuffer = _api->getEngine().getFrameBuffer();
            _oocManager->setFrameBuffer(&frameBuffer);
            _oocManager->loadBricks();
        }
}

Response BioExplorerPlugin::_getVersion() const
{
    Response response;
    response.contents = PACKAGE_VERSION;
    return response;
}

Response BioExplorerPlugin::_reset()
{
    Response response;
    auto &scene = _api->getScene();
    const auto modelDescriptors = scene.getModelDescriptors();

    for (const auto modelDescriptor : modelDescriptors)
        scene.removeModel(modelDescriptor->getModelID());

    scene.markModified();
    response.contents =
        "Removed " + std::to_string(modelDescriptors.size()) + " models";
    return response;
}

SceneInformationDetails BioExplorerPlugin::_getSceneInformation() const
{
    SceneInformationDetails sceneInfo;
    const auto &scene = _api->getScene();
    const auto &modelDescriptors = scene.getModelDescriptors();
    sceneInfo.nbModels = modelDescriptors.size();

    for (const auto modelDescriptor : modelDescriptors)
    {
        const auto &instances = modelDescriptor->getInstances();
        const auto nbInstances = instances.size();
        auto &model = modelDescriptor->getModel();
        for (const auto &spheres : model.getSpheres())
            sceneInfo.nbSpheres += nbInstances * spheres.second.size();
        for (const auto &cylinders : model.getCylinders())
            sceneInfo.nbCylinders += nbInstances * cylinders.second.size();
        for (const auto &cones : model.getCones())
            sceneInfo.nbCones += nbInstances * cones.second.size();
        for (const auto &triangleMesh : model.getTriangleMeshes())
        {
            const auto &triangle = triangleMesh.second;
            sceneInfo.nbIndices += nbInstances * triangle.indices.size();
            sceneInfo.nbVertices += nbInstances * triangle.vertices.size();
            sceneInfo.nbNormals += nbInstances * triangle.normals.size();
            sceneInfo.nbColors += nbInstances * triangle.colors.size();
        }
        sceneInfo.nbMaterials +=
            nbInstances *
            (model.getSpheres().size() + model.getCylinders().size() +
             model.getCones().size() + model.getTriangleMeshes().size());
    }
    return sceneInfo;
}

Response BioExplorerPlugin::_setGeneralSettings(
    const GeneralSettingsDetails &payload)
{
    Response response;
    try
    {
        GeneralSettings::getInstance()->setModelVisibilityOnCreation(
            payload.modelVisibilityOnCreation);
        GeneralSettings::getInstance()->setOffFolder(payload.offFolder);
        GeneralSettings::getInstance()->setLoggingEnabled(
            payload.loggingEnabled);
        PLUGIN_INFO("Setting general options for the plugin");

        response.contents = "OK";
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_removeAssembly(const AssemblyDetails &payload)
{
    Response response;
    try
    {
        auto assembly = _assemblies.find(payload.name);
        if (assembly != _assemblies.end())
            _assemblies.erase(assembly);
        else
            response.contents = "Assembly does not exist: " + payload.name;
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_addAssembly(const AssemblyDetails &payload)
{
    Response response;
    try
    {
        if (_assemblies.find(payload.name) != _assemblies.end())
            PLUGIN_THROW("Assembly already exists: " + payload.name);
        auto &scene = _api->getScene();
        AssemblyPtr assembly = AssemblyPtr(new Assembly(scene, payload));
        _assemblies[payload.name] = std::move(assembly);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_setColorScheme(
    const ColorSchemeDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, setColorScheme(payload));
}

Response BioExplorerPlugin::_setAminoAcidSequenceAsString(
    const AminoAcidSequenceAsStringDetails &payload) const
{
    if (payload.sequence.empty())
        PLUGIN_THROW("A valid sequence must be specified");
    ASSEMBLY_CALL_VOID(payload.assemblyName,
                       setAminoAcidSequenceAsString(payload));
}

Response BioExplorerPlugin::_setAminoAcidSequenceAsRanges(
    const AminoAcidSequenceAsRangesDetails &payload) const
{
    if (payload.ranges.size() % 2 != 0 || payload.ranges.size() < 2)
        PLUGIN_THROW("A valid range must be specified");
    ASSEMBLY_CALL_VOID(payload.assemblyName,
                       setAminoAcidSequenceAsRange(payload));
}

Response BioExplorerPlugin::_getAminoAcidInformation(
    const AminoAcidInformationDetails &payload) const
{
    ASSEMBLY_CALL(payload.assemblyName, getAminoAcidInformation(payload));
}

Response BioExplorerPlugin::_addRNASequence(
    const RNASequenceDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addRNASequence(payload));
}

Response BioExplorerPlugin::_addParametricMembrane(
    const ParametricMembraneDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addParametricMembrane(payload));
}

Response BioExplorerPlugin::_addMeshBasedMembrane(
    const MeshBasedMembraneDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addMeshBasedMembrane(payload));
}

Response BioExplorerPlugin::_addProtein(const ProteinDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addProtein(payload));
}

Response BioExplorerPlugin::_addGlycans(const SugarsDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addGlycans(payload));
}

Response BioExplorerPlugin::_addSugars(const SugarsDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addSugars(payload));
}

Response BioExplorerPlugin::_setAminoAcid(const AminoAcidDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, setAminoAcid(payload));
}

Response BioExplorerPlugin::_setProteinInstanceTransformation(
    const ProteinInstanceTransformationDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName,
                       setProteinInstanceTransformation(payload));
}

Response BioExplorerPlugin::_getProteinInstanceTransformation(
    const ProteinInstanceTransformationDetails &payload) const
{
    Response response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
        {
            const auto transformation =
                (*it).second->getProteinInstanceTransformation(payload);
            const auto &position = transformation.getTranslation();
            const auto &rotation = transformation.getRotation();
            std::stringstream s;
            s << "position=" << position.x << "," << position.y << ","
              << position.z << "|rotation=" << rotation.w << "," << rotation.x
              << "," << rotation.y << "," << rotation.z;
            response.contents = s.str();
        }
        else
        {
            std::stringstream msg;
            msg << "Assembly not found: " << payload.assemblyName;
            PLUGIN_ERROR(msg.str());
            response.status = false;
            response.contents = msg.str();
        }
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_exportToFile(const FileAccessDetails &payload)
{
    Response response;
    try
    {
        const Boxd bounds =
            vector_to_bounds(payload.lowBounds, payload.highBounds);
        auto &scene = _api->getScene();
        CacheLoader loader(scene);
        loader.exportToFile(payload.filename, bounds);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_importFromFile(const FileAccessDetails &payload)
{
    Response response;
    try
    {
        auto &scene = _api->getScene();
        CacheLoader loader(scene);
        const auto modelDescriptors =
            loader.importModelsFromFile(payload.filename);
        if (modelDescriptors.empty())
            PLUGIN_THROW("No models were found in " + payload.filename);
        response.contents = std::to_string(modelDescriptors.size()) +
                            " models successfully loaded";
        for (const auto modelDescriptor : modelDescriptors)
            scene.addModel(modelDescriptor);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_exportToXYZ(const FileAccessDetails &payload)
{
    Response response;
    try
    {
        auto &scene = _api->getScene();
        CacheLoader loader(scene);
        loader.exportToXYZ(payload.filename, payload.fileFormat);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_addGrid(const AddGridDetails &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO("Building Grid scene");

        auto &scene = _api->getScene();
        auto model = scene.createModel();

        const Vector3f red = {1, 0, 0};
        const Vector3f green = {0, 1, 0};
        const Vector3f blue = {0, 0, 1};
        const Vector3f grey = {0.5, 0.5, 0.5};

        PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                           static_cast<int>(MaterialShadingMode::basic)});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        props.setProperty(
            {MATERIAL_PROPERTY_CHAMELEON_MODE,
             static_cast<int>(
                 MaterialChameleonMode::undefined_chameleon_mode)});

        auto material = model->createMaterial(0, "x");
        material->setDiffuseColor(grey);
        material->setProperties(props);
        const auto position = floatsToVector3f(payload.position);

        const float m = payload.minValue;
        const float M = payload.maxValue;
        const float s = payload.steps;
        const float r = payload.radius;

        /// Grid
        for (float x = m; x <= M; x += s)
            for (float y = m; y <= M; y += s)
            {
                bool showFullGrid = true;
                if (!payload.showFullGrid)
                    showFullGrid = (fabs(x) < 0.001f || fabs(y) < 0.001f);
                if (showFullGrid)
                {
                    model->addCylinder(0, {position + Vector3f(x, y, m),
                                           position + Vector3f(x, y, M), r});
                    model->addCylinder(0, {position + Vector3f(m, x, y),
                                           position + Vector3f(M, x, y), r});
                    model->addCylinder(0, {position + Vector3f(x, m, y),
                                           position + Vector3f(x, M, y), r});
                }
            }

        if (payload.showPlanes)
        {
            // Planes
            material = model->createMaterial(1, "plane_x");
            material->setDiffuseColor(payload.useColors ? red : grey);
            material->setOpacity(payload.planeOpacity);
            material->setProperties(props);
            auto &tmx = model->getTriangleMeshes()[1];
            tmx.vertices.push_back(position + Vector3f(m, 0, m));
            tmx.vertices.push_back(position + Vector3f(M, 0, m));
            tmx.vertices.push_back(position + Vector3f(M, 0, M));
            tmx.vertices.push_back(position + Vector3f(m, 0, M));
            tmx.indices.push_back(Vector3ui(0, 1, 2));
            tmx.indices.push_back(Vector3ui(2, 3, 0));

            material = model->createMaterial(2, "plane_y");
            material->setDiffuseColor(payload.useColors ? green : grey);
            material->setOpacity(payload.planeOpacity);
            material->setProperties(props);
            auto &tmy = model->getTriangleMeshes()[2];
            tmy.vertices.push_back(position + Vector3f(m, m, 0));
            tmy.vertices.push_back(position + Vector3f(M, m, 0));
            tmy.vertices.push_back(position + Vector3f(M, M, 0));
            tmy.vertices.push_back(position + Vector3f(m, M, 0));
            tmy.indices.push_back(Vector3ui(0, 1, 2));
            tmy.indices.push_back(Vector3ui(2, 3, 0));

            material = model->createMaterial(3, "plane_z");
            material->setDiffuseColor(payload.useColors ? blue : grey);
            material->setOpacity(payload.planeOpacity);
            material->setProperties(props);
            auto &tmz = model->getTriangleMeshes()[3];
            tmz.vertices.push_back(position + Vector3f(0, m, m));
            tmz.vertices.push_back(position + Vector3f(0, m, M));
            tmz.vertices.push_back(position + Vector3f(0, M, M));
            tmz.vertices.push_back(position + Vector3f(0, M, m));
            tmz.indices.push_back(Vector3ui(0, 1, 2));
            tmz.indices.push_back(Vector3ui(2, 3, 0));
        }

        // Axis
        if (payload.showAxis)
        {
            const float l = M;
            const float smallRadius = payload.radius * 25.0;
            const float largeRadius = payload.radius * 50.0;
            const float l1 = l * 0.89;
            const float l2 = l * 0.90;

            PropertyMap props;
            props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
            props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                               static_cast<int>(MaterialShadingMode::basic)});
            props.setProperty(
                {MATERIAL_PROPERTY_CHAMELEON_MODE,
                 static_cast<int>(
                     MaterialChameleonMode::undefined_chameleon_mode)});

            // X
            material = model->createMaterial(4, "x_axis");
            material->setDiffuseColor(red);
            material->setProperties(props);

            model->addCylinder(4, {position, position + Vector3f(l1, 0, 0),
                                   smallRadius});
            model->addCone(4, {position + Vector3f(l1, 0, 0),
                               position + Vector3f(l2, 0, 0), smallRadius,
                               largeRadius});
            model->addCone(4, {position + Vector3f(l2, 0, 0),
                               position + Vector3f(M, 0, 0), largeRadius, 0});

            // Y
            material = model->createMaterial(5, "y_axis");
            material->setDiffuseColor(green);
            material->setProperties(props);

            model->addCylinder(5, {position, position + Vector3f(0, l1, 0),
                                   smallRadius});
            model->addCone(5, {position + Vector3f(0, l1, 0),
                               position + Vector3f(0, l2, 0), smallRadius,
                               largeRadius});
            model->addCone(5, {position + Vector3f(0, l2, 0),
                               position + Vector3f(0, M, 0), largeRadius, 0});

            // Z
            material = model->createMaterial(6, "z_axis");
            material->setDiffuseColor(blue);
            material->setProperties(props);

            model->addCylinder(6, {position, position + Vector3f(0, 0, l1),
                                   smallRadius});
            model->addCone(6, {position + Vector3f(0, 0, l1),
                               position + Vector3f(0, 0, l2), smallRadius,
                               largeRadius});
            model->addCone(6, {position + Vector3f(0, 0, l2),
                               position + Vector3f(0, 0, M), largeRadius, 0});

            // Origin
            model->addSphere(0, {position, smallRadius});
        }

        scene.addModel(
            std::make_shared<ModelDescriptor>(std::move(model), "Grid"));
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_addSphere(const AddSphereDetails &payload)
{
    Response response;
    try
    {
        if (payload.position.size() != 3)
            PLUGIN_THROW("Invalid number of float for position");
        if (payload.color.size() != 3)
            PLUGIN_THROW("Invalid number of float for color");

        auto &scene = _api->getScene();
        auto model = scene.createModel();

        PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                           static_cast<int>(MaterialShadingMode::electron)});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        props.setProperty(
            {MATERIAL_PROPERTY_CHAMELEON_MODE,
             static_cast<int>(
                 MaterialChameleonMode::undefined_chameleon_mode)});

        const auto color = floatsToVector3f(payload.color);
        const auto position = floatsToVector3f(payload.position);

        auto material = model->createMaterial(0, "Sphere");
        material->setDiffuseColor(color);
        material->setProperties(props);

        PLUGIN_INFO("Adding sphere " + payload.name + " to the scene");

        model->addSphere(0, {position, payload.radius});
        scene.addModel(
            std::make_shared<ModelDescriptor>(std::move(model), payload.name));
    }
    CATCH_STD_EXCEPTION()
    return response;
}

IdsDetails BioExplorerPlugin::_getModelIds() const
{
    auto &scene = _api->getScene();
    const auto &modelDescriptors = scene.getModelDescriptors();
    IdsDetails modelIds;
    for (const auto &modelDescriptor : modelDescriptors)
    {
        const auto &modelId = modelDescriptor->getModelID();
        PLUGIN_INFO("Adding model id: " + std::to_string(modelId));
        modelIds.ids.push_back(modelId);
    }
    return modelIds;
}

ModelNameDetails BioExplorerPlugin::_getModelName(
    const ModelIdDetails &payload) const
{
    auto &scene = _api->getScene();
    auto modelDescriptor = scene.getModel(payload.modelId);
    if (modelDescriptor)
    {
        ModelNameDetails modelName;
        modelName.name = modelDescriptor->getName();
        return modelName;
    }
    PLUGIN_THROW("Trying to get name from an invalid model ID: " +
                 std::to_string(payload.modelId));
}

IdsDetails BioExplorerPlugin::_getMaterialIds(const ModelIdDetails &payload)
{
    IdsDetails materialIds;
    auto &scene = _api->getScene();
    auto modelDescriptor = scene.getModel(payload.modelId);
    if (modelDescriptor)
    {
        for (const auto &material : modelDescriptor->getModel().getMaterials())
            if (material.first != BOUNDINGBOX_MATERIAL_ID &&
                material.first != SECONDARY_MODEL_MATERIAL_ID)
                materialIds.ids.push_back(material.first);
    }
    else
        PLUGIN_ERROR("Trying to get materials from an invalid model ID: " +
                     std::to_string(payload.modelId));
    return materialIds;
}

Response BioExplorerPlugin::_setMaterials(const MaterialsDetails &payload)
{
    Response response;
    try
    {
        auto &scene = _api->getScene();
        for (const auto modelId : payload.modelIds)
        {
            PLUGIN_INFO("Modifying materials on model " << modelId);
            auto modelDescriptor = scene.getModel(modelId);
            if (modelDescriptor)
            {
                size_t id = 0;
                for (const auto materialId : payload.materialIds)
                {
                    try
                    {
                        auto material =
                            modelDescriptor->getModel().getMaterial(materialId);
                        if (material)
                        {
                            if (!payload.diffuseColors.empty())
                            {
                                const size_t index = id * 3;
                                material->setDiffuseColor(
                                    {payload.diffuseColors[index],
                                     payload.diffuseColors[index + 1],
                                     payload.diffuseColors[index + 2]});
                                material->setSpecularColor(
                                    {payload.specularColors[index],
                                     payload.specularColors[index + 1],
                                     payload.specularColors[index + 2]});
                            }

                            if (!payload.specularExponents.empty())
                                material->setSpecularExponent(
                                    payload.specularExponents[id]);
                            if (!payload.reflectionIndices.empty())
                                material->setReflectionIndex(
                                    payload.reflectionIndices[id]);
                            if (!payload.opacities.empty())
                                material->setOpacity(payload.opacities[id]);
                            if (!payload.refractionIndices.empty())
                                material->setRefractionIndex(
                                    payload.refractionIndices[id]);
                            if (!payload.emissions.empty())
                                material->setEmission(payload.emissions[id]);
                            if (!payload.glossinesses.empty())
                                material->setGlossiness(
                                    payload.glossinesses[id]);
                            if (!payload.shadingModes.empty())
                                material->updateProperty(
                                    MATERIAL_PROPERTY_SHADING_MODE,
                                    payload.shadingModes[id]);
                            if (!payload.userParameters.empty())
                                material->updateProperty(
                                    MATERIAL_PROPERTY_USER_PARAMETER,
                                    static_cast<double>(
                                        payload.userParameters[id]));
                            if (!payload.chameleonModes.empty())
                                material->updateProperty(
                                    MATERIAL_PROPERTY_CHAMELEON_MODE,
                                    payload.chameleonModes[id]);

                            // This is needed to apply modifications. Changes to
                            // the material will be committed after the
                            // rendering of the current frame is completed. The
                            // false parameter is to prevent the callback to be
                            // invoked on every material, this will be done
                            // later on at a scene level
                            material->markModified(false);
                        }
                    }
                    catch (const std::runtime_error &e)
                    {
                        PLUGIN_INFO(e.what());
                    }
                    ++id;
                }
            }
            else
                PLUGIN_INFO("Model " << modelId << " is not registered");
        }
        scene.markModified(false);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

size_t BioExplorerPlugin::_attachFieldsHandler(FieldsHandlerPtr handler)
{
    auto &scene = _api->getScene();
    auto model = scene.createModel();
    const auto &spacing = Vector3f(handler->getSpacing());
    const auto &size = Vector3f(handler->getDimensions()) * spacing;
    const auto &offset = Vector3f(handler->getOffset());
    const Vector3f center{(offset + size / 2.f)};

    const size_t materialId = 0;
    auto material = model->createMaterial(materialId, "default");

    TriangleMesh box = createBox(offset, offset + size);
    model->getTriangleMeshes()[materialId] = box;
    ModelMetadata metadata;
    metadata["Center"] = std::to_string(center.x) + "," +
                         std::to_string(center.y) + "," +
                         std::to_string(center.z);
    metadata["Size"] = std::to_string(size.x) + "," + std::to_string(size.y) +
                       "," + std::to_string(size.z);
    metadata["Spacing"] = std::to_string(spacing.x) + "," +
                          std::to_string(spacing.y) + "," +
                          std::to_string(spacing.z);

    model->setSimulationHandler(handler);
    setDefaultTransferFunction(*model);

    auto modelDescriptor =
        std::make_shared<ModelDescriptor>(std::move(model), "Fields", metadata);
    scene.addModel(modelDescriptor);

    size_t modelId = modelDescriptor->getModelID();
    PLUGIN_INFO("Fields model " << modelId << " was successfully created");
    return modelId;
}

Response BioExplorerPlugin::_buildFields(const BuildFieldsDetails &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO("Building Fields from scene");
        auto &scene = _api->getScene();
        auto modelDescriptors = scene.getModelDescriptors();
        for (auto &modelDescriptor : modelDescriptors)
            if (modelDescriptor->getName() == "Fields")
                PLUGIN_THROW(
                    "BioExplorer can only handle one single fields model");

        FieldsHandlerPtr handler =
            std::make_shared<FieldsHandler>(scene, payload.voxelSize,
                                            payload.density);
        const auto modelId = _attachFieldsHandler(handler);
        response.contents = std::to_string(modelId);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_importFieldsFromFile(
    const FileAccessDetails &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO("Importing Fields from " << payload.filename);
        FieldsHandlerPtr handler =
            std::make_shared<FieldsHandler>(payload.filename);
        _attachFieldsHandler(handler);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_exportFieldsToFile(
    const ModelIdFileAccessDetails &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO("Exporting fields to " << payload.filename);
        const auto &scene = _api->getScene();
        auto modelDetails = scene.getModel(payload.modelId);
        if (modelDetails)
        {
            auto handler = modelDetails->getModel().getSimulationHandler();
            if (handler)
            {
                FieldsHandler *fieldsHandler =
                    dynamic_cast<FieldsHandler *>(handler.get());
                if (!fieldsHandler)
                    PLUGIN_THROW("Model has no fields handler");

                fieldsHandler->exportToFile(payload.filename);
            }
            else
                PLUGIN_THROW("Model has no handler");
        }
        else
            PLUGIN_THROW("Unknown model ID");
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_buildPointCloud(
    const BuildPointCloudDetails &payload)
{
    Response response;
    try
    {
        auto &scene = _api->getScene();

        const auto clipPlanes = getClippingPlanes(scene);

        auto model = scene.createModel();

        // Material
        const size_t materialId = 0;
        auto material = model->createMaterial(materialId, "Point cloud");

        PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE,
                           static_cast<int>(MaterialShadingMode::basic)});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        props.setProperty(
            {MATERIAL_PROPERTY_CHAMELEON_MODE,
             static_cast<int>(
                 MaterialChameleonMode::undefined_chameleon_mode)});
        props.setProperty({MATERIAL_PROPERTY_NODE_ID, static_cast<int>(0)});

        material->setDiffuseColor({1.f, 1.f, 1.f});
        material->updateProperties(props);

        const auto &modelDescriptors = scene.getModelDescriptors();
        for (const auto modelDescriptor : modelDescriptors)
        {
            const auto &instances = modelDescriptor->getInstances();
            for (const auto &instance : instances)
            {
                const auto &tf = instance.getTransformation();
                const auto &m = modelDescriptor->getModel();
                const auto &spheresMap = m.getSpheres();
                for (const auto &spheres : spheresMap)
                {
                    for (const auto &sphere : spheres.second)
                    {
                        const Vector3d center =
                            tf.getTranslation() +
                            tf.getRotation() * (Vector3d(sphere.center) -
                                                tf.getRotationCenter());

                        const Vector3f c = center;
                        if (isClipped(c, clipPlanes))
                            continue;

                        model->addSphere(materialId,
                                         {c, payload.radius * sphere.radius});
                    }
                }
            }
        }

        auto md =
            std::make_shared<ModelDescriptor>(std::move(model), "Point cloud");
        scene.addModel(md);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_setModelsVisibility(
    const ModelsVisibilityDetails &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO("Setting all models visibility to "
                    << (payload.visible ? "On" : "Off"));
        auto &scene = _api->getScene();
        auto &modelDescriptors = scene.getModelDescriptors();
        for (auto modelDescriptor : modelDescriptors)
            modelDescriptor->setVisible(payload.visible);
        scene.markModified();
        response.contents = "OK";
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_getOOCConfiguration() const
{
    if (!_oocManager)
        PLUGIN_THROW("Out-of-core engine is disabled");

    Response response;
    try
    {
        const auto &sceneConfiguration = _oocManager->getSceneConfiguration();
        std::stringstream s;
        s << "description=" << sceneConfiguration.description
          << "|scene_size=" << sceneConfiguration.sceneSize.x << ","
          << sceneConfiguration.sceneSize.y << ","
          << sceneConfiguration.sceneSize.z
          << "|brick_size=" << sceneConfiguration.brickSize.x << ","
          << sceneConfiguration.brickSize.y << ","
          << sceneConfiguration.brickSize.z
          << "|visible_bricks=" << _oocManager->getVisibleBricks()
          << "|update_frequency=" << _oocManager->getUpdateFrequency();
        response.contents = s.str().c_str();
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_getOOCProgress() const
{
    if (!_oocManager)
        PLUGIN_THROW("Out-of-core engine is disabled");

    Response response;
    try
    {
        const auto progress = _oocManager->getProgress();
        response.contents = std::to_string(progress);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_getOOCAverageLoadingTime() const
{
    if (!_oocManager)
        PLUGIN_THROW("Out-of-core engine is disabled");

    Response response;
    try
    {
        const auto averageLoadingTime = _oocManager->getAverageLoadingTime();
        response.contents = std::to_string(averageLoadingTime);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

#ifdef USE_PQXX
Response BioExplorerPlugin::_exportToDatabase(
    const DatabaseAccessDetails &payload)
{
    Response response;
    try
    {
        const Boxd bounds =
            vector_to_bounds(payload.lowBounds, payload.highBounds);
        auto &scene = _api->getScene();
        CacheLoader loader(scene);
        DBConnector connector(payload.connectionString, payload.schema);
        loader.exportBrickToDB(connector, payload.brickId, bounds);
    }
    CATCH_STD_EXCEPTION()
    return response;
}
#endif

extern "C" ExtensionPlugin *brayns_plugin_create(int argc, char **argv)
{
    PLUGIN_INFO(
        " _|_|_|    _|            _|_|_|_|                      _|             "
        "                           ");
    PLUGIN_INFO(
        " _|    _|        _|_|    _|        _|    _|  _|_|_|    _|    _|_|    "
        "_|  _|_|    _|_|    _|  _|_|");
    PLUGIN_INFO(
        " _|_|_|    _|  _|    _|  _|_|_|      _|_|    _|    _|  _|  _|    _|  "
        "_|_|      _|_|_|_|  _|_|    ");
    PLUGIN_INFO(
        " _|    _|  _|  _|    _|  _|        _|    _|  _|    _|  _|  _|    _|  "
        "_|        _|        _|      ");
    PLUGIN_INFO(
        " _|_|_|    _|    _|_|    _|_|_|_|  _|    _|  _|_|_|    _|    _|_|    "
        "_|          _|_|_|  _|      ");
    PLUGIN_INFO(
        "                                             _|                       "
        "                           ");
    PLUGIN_INFO(
        "                                             _|                       "
        "                           ");
    PLUGIN_INFO("Initializing BioExplorer plug-in (version " << PACKAGE_VERSION
                                                             << ")");
#ifdef USE_CGAL
    PLUGIN_INFO("- CGAL module loaded");
#endif
#ifdef USE_PQXX
    PLUGIN_INFO("- Postgresql module loaded");
#endif

    return new BioExplorerPlugin(argc, argv);
}

} // namespace bioexplorer
