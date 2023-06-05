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

#include "BioExplorerPlugin.h"

#include <Version.h>

#include <science/common/Assembly.h>
#include <science/common/GeneralSettings.h>
#include <science/common/Logs.h>
#include <science/common/Utils.h>
#include <science/io/CacheLoader.h>
#include <science/io/OOCManager.h>
#include <science/morphologies/SpikeSimulationHandler.h>

#include <platform/core/common/ActionInterface.h>
#include <platform/core/common/scene/ClipPlane.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>

#ifdef USE_OPTIX6
#include <BioExplorer_generated_Density.cu.ptx.h>
#include <BioExplorer_generated_Fields.cu.ptx.h>
#include <BioExplorer_generated_Golgi.cu.ptx.h>
#include <BioExplorer_generated_PathTracing.cu.ptx.h>
#include <BioExplorer_generated_Voxel.cu.ptx.h>
#include <platform/engines/optix6/OptiXContext.h>
#endif

namespace bioexplorer
{
using namespace common;
using namespace io;
using namespace db;

#ifdef USE_VASCULATURE
using namespace vasculature;
#endif

const std::string PLUGIN_API_PREFIX = "be-";
const std::string RENDERER_GOLGI_STYLE = "bio_explorer_golgi_style";
const std::string RENDERER_DENSITY = "bio_explorer_density";
const std::string RENDERER_FIELDS = "bio_explorer_fields";
const std::string RENDERER_PATH_TRACING = "bio_explorer_path_tracing";
const std::string RENDERER_VOXEL = "bio_explorer_voxel";

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

Boxd vector_to_bounds(const doubles &lowBounds, const doubles &highBounds)
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

void _addBioExplorerVoxelRenderer(Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_VOXEL);
    PropertyMap properties;
    properties.setProperty({"alphaCorrection", 0.5, 0.001, 1., {"Alpha correction"}});
    properties.setProperty({"simulationThreshold", 0., 0., 1., {"Simulation threshold"}});
    properties.setProperty({"mainExposure", 1.0, 0.01, 10.0, {"Exposure"}});
    properties.setProperty({"epsilonFactor", 1.0, 1.0, 1000.0, {"Epsilon factor"}});
    engine.addRendererType(RENDERER_VOXEL, properties);
}

void _addBioExplorerFieldsRenderer(Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_FIELDS);
    PropertyMap properties;
    properties.setProperty({"mainExposure", 1., 1., 10., {"Exposure"}});
    properties.setProperty({"useHardwareRandomizer", false, {"Use hardware accelerated randomizer"}});
    properties.setProperty({"minRayStep", 0.00001, 0.00001, 1.0, {"Smallest ray step"}});
    properties.setProperty({"nbRaySteps", 8, 1, 2048, {"Number of ray marching steps"}});
    properties.setProperty({"nbRayRefinementSteps", 8, 1, 1000, {"Number of ray marching refinement steps"}});
    properties.setProperty({"cutoff", 2000.0, 0.0, 1e5, {"cutoff"}});
    properties.setProperty({"alphaCorrection", 1.0, 0.001, 1.0, {"Alpha correction"}});
    engine.addRendererType(RENDERER_FIELDS, properties);
}

void _addBioExplorerDensityRenderer(Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_DENSITY);
    PropertyMap properties;
    properties.setProperty({"mainExposure", 1.5, 1., 10., {"Exposure"}});
    properties.setProperty({"rayStep", 2.0, 1.0, 1024.0, {"Ray marching step"}});
    properties.setProperty({"samplesPerFrame", 16, 1, 256, {"Samples per frame"}});
    properties.setProperty({"rayLength", 1e6, 1e-3, 1e6, {"Ray length"}});
    properties.setProperty({"farPlane", 1000.0, 1.0, 1e6, {"Far plane"}});
    properties.setProperty({"alphaCorrection", 1.0, 0.001, 1.0, {"Alpha correction"}});
    engine.addRendererType(RENDERER_DENSITY, properties);
}

void _addBioExplorerPathTracingRenderer(Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_PATH_TRACING);
    PropertyMap properties;
    properties.setProperty({"mainExposure", 1., 0.1, 10., {"Exposure"}});
    properties.setProperty({"useHardwareRandomizer", false, {"Use hardware accelerated randomizer"}});
    properties.setProperty({"showBackground", false, {"Show background"}});
    properties.setProperty({"aoStrength", 1.0, 0.0001, 10.0, {"Sample search strength"}});
    properties.setProperty({"aoDistance", 1e6, 0.1, 1e6, {"Sample search distance"}});
    engine.addRendererType(RENDERER_PATH_TRACING, properties);
}

void _addBioExplorerGolgiStyleRenderer(Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_GOLGI_STYLE);
    PropertyMap properties;
    properties.setProperty({"exponent", 5., 0.1, 10., {"Exponent"}});
    properties.setProperty({"inverse", false, {"Inverse"}});
    engine.addRendererType(RENDERER_GOLGI_STYLE, properties);
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

    registry.registerLoader(std::make_unique<CacheLoader>(scene, CacheLoader::getCLIProperties()));

    if (actionInterface)
    {
        std::string endPoint = PLUGIN_API_PREFIX + "get-version";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<Response>(endPoint, [&]() { return _getVersion(); });

        endPoint = PLUGIN_API_PREFIX + "get-scene-information";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<SceneInformationDetails>(endPoint, [&]() { return _getSceneInformation(); });

        endPoint = PLUGIN_API_PREFIX + "set-general-settings";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<GeneralSettingsDetails, Response>(endPoint,
                                                                           [&](const GeneralSettingsDetails &payload)
                                                                           { return _setGeneralSettings(payload); });

        endPoint = PLUGIN_API_PREFIX + "reset-scene";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<Response>(endPoint, [&]() { return _resetScene(); });

        endPoint = PLUGIN_API_PREFIX + "reset-camera";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<Response>(endPoint, [&]() { return _resetCamera(); });

        endPoint = PLUGIN_API_PREFIX + "set-focus-on";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<FocusOnDetails, Response>(endPoint, [&](const FocusOnDetails &payload)
                                                                   { return _setFocusOn(payload); });

        endPoint = PLUGIN_API_PREFIX + "remove-assembly";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AssemblyDetails, Response>(endPoint, [&](const AssemblyDetails &payload)
                                                                    { return _removeAssembly(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-assembly";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AssemblyDetails, Response>(endPoint, [&](const AssemblyDetails &payload)
                                                                    { return _addAssembly(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-protein-color-scheme";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<ProteinColorSchemeDetails, Response>(
            endPoint, [&](const ProteinColorSchemeDetails &payload) { return _setProteinColorScheme(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-protein-amino-acid-sequence-as-string";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AminoAcidSequenceAsStringDetails, Response>(
            endPoint,
            [&](const AminoAcidSequenceAsStringDetails &payload) { return _setAminoAcidSequenceAsString(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-protein-amino-acid-sequence-as-ranges";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AminoAcidSequenceAsRangesDetails, Response>(
            endPoint,
            [&](const AminoAcidSequenceAsRangesDetails &payload) { return _setAminoAcidSequenceAsRanges(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-protein-amino-acid-information";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AminoAcidInformationDetails, Response>(
            endPoint, [&](const AminoAcidInformationDetails &payload) { return _getAminoAcidInformation(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-protein-amino-acid";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AminoAcidDetails, Response>(endPoint, [&](const AminoAcidDetails &payload)
                                                                     { return _setAminoAcid(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-protein-instance-transformation";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<ProteinInstanceTransformationDetails, Response>(
            endPoint, [&](const ProteinInstanceTransformationDetails &payload)
            { return _setProteinInstanceTransformation(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-protein-instance-transformation";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<ProteinInstanceTransformationDetails, Response>(
            endPoint, [&](const ProteinInstanceTransformationDetails &payload)
            { return _getProteinInstanceTransformation(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-rna-sequence";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<RNASequenceDetails, Response>(endPoint, [&](const RNASequenceDetails &payload)
                                                                       { return _addRNASequence(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-membrane";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<MembraneDetails, Response>(endPoint, [&](const MembraneDetails &payload)
                                                                    { return _addMembrane(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-protein";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<ProteinDetails, Response>(endPoint, [&](const ProteinDetails &payload)
                                                                   { return _addProtein(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-glycan";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<SugarDetails, Response>(endPoint, [&](const SugarDetails &payload)
                                                                 { return _addGlycan(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-sugar";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<SugarDetails, Response>(endPoint, [&](const SugarDetails &payload)
                                                                 { return _addSugar(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-enzyme-reaction";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<EnzymeReactionDetails, Response>(endPoint,
                                                                          [&](const EnzymeReactionDetails &payload)
                                                                          { return _addEnzymeReaction(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-enzyme-reaction-progress";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<EnzymeReactionProgressDetails, Response>(
            endPoint,
            [&](const EnzymeReactionProgressDetails &payload) { return _setEnzymeReactionProgress(payload); });

        endPoint = PLUGIN_API_PREFIX + "export-to-file";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<FileAccessDetails, Response>(endPoint, [&](const FileAccessDetails &payload)
                                                                      { return _exportToFile(payload); });

        endPoint = PLUGIN_API_PREFIX + "import-from-file";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<FileAccessDetails, Response>(endPoint, [&](const FileAccessDetails &payload)
                                                                      { return _importFromFile(payload); });

        endPoint = PLUGIN_API_PREFIX + "export-to-xyz";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<FileAccessDetails, Response>(endPoint, [&](const FileAccessDetails &payload)
                                                                      { return _exportToXYZ(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-grid";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AddGridDetails, Response>(endPoint, [&](const AddGridDetails &payload)
                                                                   { return _addGrid(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-spheres";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AddSpheresDetails, Response>(endPoint, [&](const AddSpheresDetails &payload)
                                                                      { return _addSpheres(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-cone";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AddConeDetails, Response>(endPoint, [&](const AddConeDetails &payload)
                                                                   { return _addCone(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-bounding-box";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AddBoundingBoxDetails, Response>(endPoint,
                                                                          [&](const AddBoundingBoxDetails &payload)
                                                                          { return _addBoundingBox(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-box";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AddBoxDetails, Response>(endPoint, [&](const AddBoxDetails &payload)
                                                                  { return _addBox(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-streamlines";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<AddStreamlinesDetails, Response>(endPoint,
                                                                          [&](const AddStreamlinesDetails &payload)
                                                                          { return _addStreamlines(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-model-ids";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<IdsDetails>(endPoint, [&]() -> IdsDetails { return _getModelIds(); });

        endPoint = PLUGIN_API_PREFIX + "get-model-instances";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<ModelIdDetails, IdsDetails>(endPoint,
                                                                     [&](const ModelIdDetails &payload) -> IdsDetails
                                                                     { return _getModelInstances(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-model-name";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<ModelIdDetails, NameDetails>(endPoint,
                                                                      [&](const ModelIdDetails &payload) -> NameDetails
                                                                      { return _getModelName(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-materials";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<MaterialsDetails, Response>(endPoint, [&](const MaterialsDetails &payload)
                                                                     { return _setMaterials(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-material-ids";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<ModelIdDetails, IdsDetails>(endPoint,
                                                                     [&](const ModelIdDetails &payload) -> IdsDetails
                                                                     { return _getMaterialIds(payload); });

        endPoint = PLUGIN_API_PREFIX + "build-fields";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<BuildFieldsDetails, Response>(endPoint, [&](const BuildFieldsDetails &payload)
                                                                       { return _buildFields(payload); });

        endPoint = PLUGIN_API_PREFIX + "export-fields-to-file";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<ModelIdFileAccessDetails, Response>(
            endPoint, [&](const ModelIdFileAccessDetails &payload) { return _exportFieldsToFile(payload); });

        endPoint = PLUGIN_API_PREFIX + "import-fields-from-file";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<FileAccessDetails, Response>(endPoint, [&](const FileAccessDetails &payload)
                                                                      { return _importFieldsFromFile(payload); });

        endPoint = PLUGIN_API_PREFIX + "build-point-cloud";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<BuildPointCloudDetails, Response>(endPoint,
                                                                           [&](const BuildPointCloudDetails &payload)
                                                                           { return _buildPointCloud(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-models-visibility";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<ModelsVisibilityDetails, Response>(endPoint,
                                                                            [&](const ModelsVisibilityDetails &payload)
                                                                            { return _setModelsVisibility(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-out-of-core-configuration";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<Response>(endPoint, [&]() { return _getOOCConfiguration(); });

        endPoint = PLUGIN_API_PREFIX + "get-out-of-core-progress";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<Response>(endPoint, [&]() { return _getOOCProgress(); });

        endPoint = PLUGIN_API_PREFIX + "get-out-of-core-average-loading-time";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<Response>(endPoint, [&]() { return _getOOCAverageLoadingTime(); });

        endPoint = PLUGIN_API_PREFIX + "inspect-protein";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<InspectionDetails, ProteinInspectionDetails>(
            endPoint, [&](const InspectionDetails &payload) { return _inspectProtein(payload); });

        endPoint = PLUGIN_API_PREFIX + "export-to-database";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<DatabaseAccessDetails, Response>(endPoint,
                                                                          [&](const DatabaseAccessDetails &payload)
                                                                          { return _exportBrickToDatabase(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-atlas";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<AtlasDetails, Response>(endPoint, [&](const AtlasDetails &payload)
                                                                            { return _addAtlas(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-vasculature";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<VasculatureDetails, Response>(endPoint,
                                                                                  [&](const VasculatureDetails &payload)
                                                                                  { return _addVasculature(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-vasculature-info";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<NameDetails, Response>(endPoint,
                                                                [&](const NameDetails &payload) -> Response
                                                                { return _getVasculatureInfo(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-vasculature-report";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<VasculatureReportDetails, Response>(
            endPoint, [&](const VasculatureReportDetails &payload) { return _setVasculatureReport(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-vasculature-radius-report";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        actionInterface->registerRequest<VasculatureRadiusReportDetails, Response>(
            endPoint,
            [&](const VasculatureRadiusReportDetails &details) { return _setVasculatureRadiusReport(details); });
        endPoint = PLUGIN_API_PREFIX + "add-astrocytes";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<AstrocytesDetails, Response>(endPoint,
                                                                                 [&](const AstrocytesDetails &payload)
                                                                                 { return _addAstrocytes(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-neurons";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<NeuronsDetails, Response>(endPoint,
                                                                              [&](const NeuronsDetails &payload)
                                                                              { return _addNeurons(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-neuron-section-points";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<NeuronIdSectionIdDetails, NeuronPointsDetails>(
            endPoint, [&](const NeuronIdSectionIdDetails &payload) { return _getNeuronSectionPoints(payload); });

        endPoint = PLUGIN_API_PREFIX + "get-neuron-varicosities";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<NeuronIdDetails, NeuronPointsDetails>(
            endPoint, [&](const NeuronIdDetails &payload) { return _getNeuronVaricosities(payload); });

        endPoint = PLUGIN_API_PREFIX + "set-spike-report-visualization-settings";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<SpikeReportVisualizationSettingsDetails, Response>(
            endPoint,
            [&](const SpikeReportVisualizationSettingsDetails &s) { return _setSpikeReportVisualizationSettings(s); });

        endPoint = PLUGIN_API_PREFIX + "add-white-matter";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<WhiteMatterDetails, Response>(endPoint,
                                                                                  [&](const WhiteMatterDetails &payload)
                                                                                  { return _addWhiteMatter(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-synapses";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<SynapsesDetails, Response>(endPoint,
                                                                               [&](const SynapsesDetails &payload)
                                                                               { return _addSynapses(payload); });

        endPoint = PLUGIN_API_PREFIX + "add-synapse-efficacy";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<SynapseEfficacyDetails, Response>(
            endPoint, [&](const SynapseEfficacyDetails &payload) { return _addSynapseEfficacy(payload); });

        endPoint = PLUGIN_API_PREFIX + "look-at";
        PLUGIN_REGISTER_ENDPOINT(endPoint);
        _api->getActionInterface()->registerRequest<LookAtDetails, LookAtResponseDetails>(
            endPoint, [&](const LookAtDetails &payload) { return _lookAt(payload); });
    }

    auto &params = engine.getParametersManager().getApplicationParameters();
    const auto &engineName = params.getEngine();
#ifdef USE_OPTIX6
    if (engineName == ENGINE_OPTIX_6)
    {
        _createOptiXRenderers();
        _createRenderers();
    }
#endif
    if (engineName == ENGINE_OSPRAY)
        _createRenderers();

    // Database
    try
    {
        auto &dbConnector = DBConnector::getInstance();
        dbConnector.init(_commandLineArguments);
    }
    catch (const std::runtime_error &e)
    {
        PLUGIN_ERROR(e.what());
    }

    // Out-of-core
    if (_commandLineArguments.find(ARG_OOC_ENABLED) != _commandLineArguments.end())
    {
        _oocManager = OOCManagerPtr(new OOCManager(scene, camera, _commandLineArguments));
        if (_oocManager->getShowGrid())
        {
            AddGridDetails grid;
            const auto &sceneConfiguration = _oocManager->getSceneConfiguration();
            const auto sceneSize = sceneConfiguration.sceneSize.x;
            const auto brickSize = sceneConfiguration.brickSize.x;
            grid.position = {-brickSize / 2.0, -brickSize / 2.0, -brickSize / 2.0};
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

#ifdef USE_OPTIX6
void BioExplorerPlugin::_createOptiXRenderers()
{
    std::map<std::string, std::string> renderers = {
        {RENDERER_GOLGI_STYLE, BioExplorer_generated_Golgi_cu_ptx},
        {RENDERER_DENSITY, BioExplorer_generated_Density_cu_ptx},
        {RENDERER_FIELDS, BioExplorer_generated_Fields_cu_ptx},
        {RENDERER_PATH_TRACING, BioExplorer_generated_PathTracing_cu_ptx},
        {RENDERER_VOXEL, BioExplorer_generated_Voxel_cu_ptx},
    };
    OptiXContext &context = OptiXContext::get();
    for (const auto &renderer : renderers)
    {
        PLUGIN_REGISTER_RENDERER(renderer.first);
        const std::string ptx = renderer.second;

        auto osp = std::make_shared<OptixShaderProgram>();
        osp->closest_hit = context.getOptixContext()->createProgramFromPTXString(ptx, "closest_hit_radiance");
        osp->any_hit = context.getOptixContext()->createProgramFromPTXString(ptx, "any_hit_shadow");

        context.addRenderer(renderer.first, osp);
    }
}
#endif

void BioExplorerPlugin::_createRenderers()
{
    // Renderers
    auto &engine = _api->getEngine();
    _addBioExplorerVoxelRenderer(engine);
    _addBioExplorerFieldsRenderer(engine);
    _addBioExplorerDensityRenderer(engine);
    _addBioExplorerPathTracingRenderer(engine);
    _addBioExplorerGolgiStyleRenderer(engine);
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
            PLUGIN_INFO(1, "Starting Out-Of-Core manager");
            auto &frameBuffer = _api->getEngine().getFrameBuffer();
            _oocManager->setFrameBuffer(&frameBuffer);
            _oocManager->loadBricks();
        }
}

Response BioExplorerPlugin::_getVersion() const
{
    Response response;
    response.contents = PACKAGE_VERSION_STRING;
    return response;
}

Response BioExplorerPlugin::_resetScene()
{
    Response response;
    auto &scene = _api->getScene();
    const auto modelDescriptors = scene.getModelDescriptors();

    for (const auto modelDescriptor : modelDescriptors)
        scene.removeModel(modelDescriptor->getModelID());

    scene.markModified();

    _assemblies.clear();

    response.contents = "Removed " + std::to_string(modelDescriptors.size()) + " models";
    return response;
}

Response BioExplorerPlugin::_resetCamera()
{
    Response response;
    const auto &scene = _api->getScene();
    auto &camera = _api->getCamera();

    const auto &modelDescriptors = scene.getModelDescriptors();
    if (modelDescriptors.empty())
    {
        response.status = false;
        response.contents = "Cannot reset camera on an empty scene";
        return response;
    }

    Boxd aabb;
    for (const auto modelDescriptor : scene.getModelDescriptors())
    {
        const auto &modelBounds = modelDescriptor->getModel().getBounds();
        Transformation modelTransformation;
        modelTransformation.setTranslation(modelBounds.getCenter());

        const auto modelHalfSize = modelBounds.getSize() / 2.0;

        Transformation finalTransformation = modelTransformation * modelDescriptor->getTransformation();
        aabb.merge(finalTransformation.getTranslation() - modelHalfSize);
        aabb.merge(finalTransformation.getTranslation() + modelHalfSize);
        for (const auto &instance : modelDescriptor->getInstances())
        {
            finalTransformation = modelTransformation * instance.getTransformation();
            aabb.merge(finalTransformation.getTranslation() - modelHalfSize);
            aabb.merge(finalTransformation.getTranslation() + modelHalfSize);
        }
    }

    const auto size = aabb.getSize();
    const double diag = 1.6 * std::max(std::max(size.x, size.y), size.z);
    camera.setPosition(aabb.getCenter() + Vector3d(0.0, 0.0, diag));
    camera.setTarget(aabb.getCenter());
    camera.setOrientation(safeQuatlookAt(Vector3d(0.0, 0.0, -1.0)));
    return response;
}

Response BioExplorerPlugin::_setFocusOn(const FocusOnDetails &payload)
{
    Response response;
    try
    {
        const auto &scene = _api->getScene();

        auto &camera = _api->getCamera();
        auto modelDescriptor = scene.getModel(payload.modelId);
        if (!modelDescriptor)
            PLUGIN_THROW("Invalid model Id");

        const auto &instances = modelDescriptor->getInstances();
        if (payload.instanceId >= instances.size())
            PLUGIN_THROW("Invalid instance Id");

        const auto &instance = instances[payload.instanceId];
        const auto &transformation = instance.getTransformation();

        double distance = payload.distance;
        if (distance == 0.0)
        {
            const auto &aabb = modelDescriptor->getBounds();
            const auto size = aabb.getSize();
            distance = 3.0 * std::max(std::max(size.x, size.y), size.z);
        }

        const auto direction = -normalize(doublesToVector3d(payload.direction));
        const auto &translation = transformation.getTranslation();
        camera.setPosition(translation - direction * distance);
        camera.setTarget(translation);
        camera.setOrientation(safeQuatlookAt(direction));
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
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
        sceneInfo.nbMaterials += nbInstances * (model.getSpheres().size() + model.getCylinders().size() +
                                                model.getCones().size() + model.getTriangleMeshes().size());
    }
    return sceneInfo;
}

Response BioExplorerPlugin::_setGeneralSettings(const GeneralSettingsDetails &payload)
{
    Response response;
    try
    {
        auto instance = GeneralSettings::getInstance();
        instance->setModelVisibilityOnCreation(payload.modelVisibilityOnCreation);
        instance->setMeshFolder(payload.meshFolder);
        instance->setLoggingLevel(payload.loggingLevel);
        instance->setDBLoggingLevel(payload.databaseLoggingLevel);
        instance->setV1Compatibility(payload.v1Compatibility);
        PLUGIN_INFO(3, "Setting general options for the plugin");

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

Response BioExplorerPlugin::_setProteinColorScheme(const ProteinColorSchemeDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, setProteinColorScheme(payload));
}

Response BioExplorerPlugin::_setAminoAcidSequenceAsString(const AminoAcidSequenceAsStringDetails &payload) const
{
    if (payload.sequence.empty())
        PLUGIN_THROW("A valid sequence must be specified");
    ASSEMBLY_CALL_VOID(payload.assemblyName, setAminoAcidSequenceAsString(payload));
}

Response BioExplorerPlugin::_setAminoAcidSequenceAsRanges(const AminoAcidSequenceAsRangesDetails &payload) const
{
    if (payload.ranges.size() % 2 != 0 || payload.ranges.size() < 2)
        PLUGIN_THROW("A valid range must be specified");
    ASSEMBLY_CALL_VOID(payload.assemblyName, setAminoAcidSequenceAsRange(payload));
}

Response BioExplorerPlugin::_getAminoAcidInformation(const AminoAcidInformationDetails &payload) const
{
    ASSEMBLY_CALL(payload.assemblyName, getAminoAcidInformation(payload));
}

Response BioExplorerPlugin::_addRNASequence(const RNASequenceDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addRNASequence(payload));
}

Response BioExplorerPlugin::_addMembrane(const MembraneDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addMembrane(payload));
}

Response BioExplorerPlugin::_addProtein(const ProteinDetails &payload) const
{
    AssemblyConstraints constraints;
    const auto values = split(payload.constraints, CONTENTS_DELIMITER);
    for (const auto &value : values)
    {
        const auto assemblyConstraintType =
            (value[0] == '+' ? AssemblyConstraintType::inside : AssemblyConstraintType::outside);
        auto assemblyName = value;
        assemblyName.erase(0, 1);

        const auto it = _assemblies.find(assemblyName);
        if (it != _assemblies.end())
            constraints.push_back(AssemblyConstraint(assemblyConstraintType, (*it).second));
        else
            PLUGIN_THROW("Unknown assembly specified in the location constraints: " + assemblyName);
    }

    ASSEMBLY_CALL_VOID(payload.assemblyName, addProtein(payload, constraints));
}

Response BioExplorerPlugin::_addGlycan(const SugarDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addGlycan(payload));
}

Response BioExplorerPlugin::_addSugar(const SugarDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addSugar(payload));
}

Response BioExplorerPlugin::_addEnzymeReaction(const EnzymeReactionDetails &payload) const
{
    Response response;
    try
    {
        AssemblyPtr enzymeAssembly{nullptr};
        ProteinPtr enzyme{nullptr};

        const auto substrateNames = split(payload.substrateNames);
        const auto productNames = split(payload.productNames);
        Proteins substrates(substrateNames.size());
        Proteins products(productNames.size());

        uint64_t index = 0;
        for (const auto &substrateName : substrateNames)
        {
            for (auto &assembly : _assemblies)
            {
                auto substrate = assembly.second->getProtein(substrateName);
                if (substrate)
                {
                    substrates[index] = substrate;
                    break;
                }
            }
            ++index;
        }

        index = 0;
        for (const auto &productName : productNames)
        {
            for (auto &assembly : _assemblies)
            {
                auto product = assembly.second->getProtein(productName);
                if (product)
                {
                    products[index] = product;
                    break;
                }
            }
            ++index;
        }

        for (auto &assembly : _assemblies)
        {
            enzyme = assembly.second->getProtein(payload.enzymeName);
            if (enzyme)
            {
                enzymeAssembly = assembly.second;
                break;
            }
        }

        if (!enzyme)
            PLUGIN_THROW("Enzyme " + payload.enzymeName + " could not be found in scene");
        if (substrates.size() != substrateNames.size())
            PLUGIN_THROW("Some substrates could not be found in scene");
        if (products.size() != productNames.size())
            PLUGIN_THROW("Some products could not be found in scene");

        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
            (*it).second->addEnzymeReaction(payload, enzymeAssembly, enzyme, substrates, products);
        else
            PLUGIN_THROW("Assembly " + payload.assemblyName + " not found");
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        response.contents = e.what();
    }
    return response;
}

Response BioExplorerPlugin::_setEnzymeReactionProgress(const EnzymeReactionProgressDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, setEnzymeReactionProgress(payload));
}

Response BioExplorerPlugin::_setAminoAcid(const AminoAcidDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, setAminoAcid(payload));
}

Response BioExplorerPlugin::_setProteinInstanceTransformation(const ProteinInstanceTransformationDetails &payload) const
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, setProteinInstanceTransformation(payload));
}

Response BioExplorerPlugin::_getProteinInstanceTransformation(const ProteinInstanceTransformationDetails &payload) const
{
    Response response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
        {
            const auto transformation = (*it).second->getProteinInstanceTransformation(payload);
            const auto &position = transformation.getTranslation();
            const auto &rotation = transformation.getRotation();
            std::stringstream s;
            s << "position=" << position.x << "," << position.y << "," << position.z << "|rotation=" << rotation.w
              << "," << rotation.x << "," << rotation.y << "," << rotation.z;
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
        const Boxd bounds = vector_to_bounds(payload.lowBounds, payload.highBounds);
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
        const auto modelDescriptors = loader.importModelsFromFile(payload.filename);
        if (modelDescriptors.empty())
            PLUGIN_THROW("No models were found in " + payload.filename);
        response.contents = std::to_string(modelDescriptors.size()) + " models successfully loaded";
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
        PLUGIN_INFO(3, "Building Grid scene");

        auto &scene = _api->getScene();
        auto model = scene.createModel();

        const Vector3d red = {1, 0, 0};
        const Vector3d green = {0, 1, 0};
        const Vector3d blue = {0, 0, 1};
        const Vector3d grey = {0.5, 0.5, 0.5};

        PropertyMap props;
        auto material = model->createMaterial(0, "x");
        material->setDiffuseColor(grey);
        material->setProperties(props);
        const auto position = doublesToVector3d(payload.position);

        const double m = payload.minValue;
        const double M = payload.maxValue;
        const double s = payload.steps;
        const float r = static_cast<float>(payload.radius);

        /// Grid
        for (double x = m; x <= M; x += s)
            for (double y = m; y <= M; y += s)
            {
                bool showFullGrid = true;
                if (!payload.showFullGrid)
                    showFullGrid = (fabs(x) < 0.001f || fabs(y) < 0.001f);
                if (showFullGrid)
                {
                    model->addCylinder(0, {position + Vector3d(x, y, m), position + Vector3d(x, y, M), r});
                    model->addCylinder(0, {position + Vector3d(m, x, y), position + Vector3d(M, x, y), r});
                    model->addCylinder(0, {position + Vector3d(x, m, y), position + Vector3d(x, M, y), r});
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
            tmx.vertices.push_back(position + Vector3d(m, 0, m));
            tmx.vertices.push_back(position + Vector3d(M, 0, m));
            tmx.vertices.push_back(position + Vector3d(M, 0, M));
            tmx.vertices.push_back(position + Vector3d(m, 0, M));
            tmx.indices.push_back(Vector3ui(0, 1, 2));
            tmx.indices.push_back(Vector3ui(2, 3, 0));

            material = model->createMaterial(2, "plane_y");
            material->setDiffuseColor(payload.useColors ? green : grey);
            material->setOpacity(payload.planeOpacity);
            material->setProperties(props);
            auto &tmy = model->getTriangleMeshes()[2];
            tmy.vertices.push_back(position + Vector3d(m, m, 0));
            tmy.vertices.push_back(position + Vector3d(M, m, 0));
            tmy.vertices.push_back(position + Vector3d(M, M, 0));
            tmy.vertices.push_back(position + Vector3d(m, M, 0));
            tmy.indices.push_back(Vector3ui(0, 1, 2));
            tmy.indices.push_back(Vector3ui(2, 3, 0));

            material = model->createMaterial(3, "plane_z");
            material->setDiffuseColor(payload.useColors ? blue : grey);
            material->setOpacity(payload.planeOpacity);
            material->setProperties(props);
            auto &tmz = model->getTriangleMeshes()[3];
            tmz.vertices.push_back(position + Vector3d(0, m, m));
            tmz.vertices.push_back(position + Vector3d(0, m, M));
            tmz.vertices.push_back(position + Vector3d(0, M, M));
            tmz.vertices.push_back(position + Vector3d(0, M, m));
            tmz.indices.push_back(Vector3ui(0, 1, 2));
            tmz.indices.push_back(Vector3ui(2, 3, 0));
        }

        // Axis
        if (payload.showAxis)
        {
            const double l = M;
            const float smallRadius = r * 25.0;
            const float largeRadius = r * 50.0;
            const double l1 = l * 0.89;
            const double l2 = l * 0.90;

            // X
            material = model->createMaterial(4, "x_axis");
            material->setDiffuseColor(red);
            material->setProperties(props);

            model->addCylinder(4, {position, position + Vector3d(l1, 0, 0), smallRadius});
            model->addCone(4, {position + Vector3d(l1, 0, 0), position + Vector3d(l2, 0, 0), smallRadius, largeRadius});
            model->addCone(4, {position + Vector3d(l2, 0, 0), position + Vector3d(M, 0, 0), largeRadius, 0});

            // Y
            material = model->createMaterial(5, "y_axis");
            material->setDiffuseColor(green);

            model->addCylinder(5, {position, position + Vector3d(0, l1, 0), smallRadius});
            model->addCone(5, {position + Vector3d(0, l1, 0), position + Vector3d(0, l2, 0), smallRadius, largeRadius});
            model->addCone(5, {position + Vector3d(0, l2, 0), position + Vector3d(0, M, 0), largeRadius, 0});

            // Z
            material = model->createMaterial(6, "z_axis");
            material->setDiffuseColor(blue);

            model->addCylinder(6, {position, position + Vector3d(0, 0, l1), smallRadius});
            model->addCone(6, {position + Vector3d(0, 0, l1), position + Vector3d(0, 0, l2), smallRadius, largeRadius});
            model->addCone(6, {position + Vector3d(0, 0, l2), position + Vector3d(0, 0, M), largeRadius, 0});

            // Origin
            model->addSphere(0, {position, smallRadius});
        }

        scene.addModel(std::make_shared<ModelDescriptor>(std::move(model), "Grid"));
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_addSpheres(const AddSpheresDetails &payload)
{
    Response response;
    try
    {
        if (payload.positions.size() % 3 != 0)
            PLUGIN_THROW("Invalid number of doubles for positions");
        if (payload.positions.size() / 3 != payload.radii.size())
            PLUGIN_THROW("Invalid number of radii");
        if (payload.color.size() != 3)
            PLUGIN_THROW("Invalid number of doubles for color");

        auto &scene = _api->getScene();
        auto model = scene.createModel();

        const auto color = doublesToVector3d(payload.color);

        const size_t materialId = 0;
        auto material = model->createMaterial(materialId, "Spheres");
        material->setDiffuseColor(color);
        material->setOpacity(payload.opacity);

        PLUGIN_INFO(3, "Adding spheres " + payload.name + " to the scene");

        for (uint64_t i = 0; i < payload.radii.size(); ++i)
        {
            const auto position =
                Vector3d(payload.positions[i * 3], payload.positions[i * 3 + 1], payload.positions[i * 3 + 2]);

            model->addSphere(materialId, {position, static_cast<float>(payload.radii[i])});
        }
        model->updateBounds();
        scene.addModel(std::make_shared<ModelDescriptor>(std::move(model), payload.name));
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_addCone(const AddConeDetails &payload)
{
    Response response;
    try
    {
        if (payload.origin.size() != 3)
            PLUGIN_THROW("Invalid number of double for origin");
        if (payload.target.size() != 3)
            PLUGIN_THROW("Invalid number of double for target");
        if (payload.color.size() != 3)
            PLUGIN_THROW("Invalid number of double for color");

        auto &scene = _api->getScene();
        auto model = scene.createModel();

        const auto color = doublesToVector3d(payload.color);
        const auto origin = doublesToVector3d(payload.origin);
        const auto target = doublesToVector3d(payload.target);

        auto material = model->createMaterial(0, "Cone");
        material->setDiffuseColor(color);
        material->setOpacity(payload.opacity);

        PLUGIN_INFO(3, "Adding cone " + payload.name + " to the scene");

        if (payload.originRadius == payload.targetRadius)
            model->addCylinder(0, {origin, target, static_cast<float>(payload.originRadius)});
        else
            model->addCone(0, {origin, target, static_cast<float>(payload.originRadius),
                               static_cast<float>(payload.targetRadius)});
        scene.addModel(std::make_shared<ModelDescriptor>(std::move(model), payload.name));
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_addBoundingBox(const AddBoundingBoxDetails &payload)
{
    Response response;
    try
    {
        if (payload.bottomLeft.size() != 3)
            PLUGIN_THROW("Invalid number of double for bottom left corner");
        if (payload.topRight.size() != 3)
            PLUGIN_THROW("Invalid number of double for top right corner");
        if (payload.color.size() != 3)
            PLUGIN_THROW("Invalid number of double for color");

        auto &scene = _api->getScene();
        auto model = scene.createModel();

        const auto color = doublesToVector3d(payload.color);
        auto material = model->createMaterial(0, "BoundingBox");
        material->setDiffuseColor(color);

        PLUGIN_INFO(3, "Adding bounding box " + payload.name + " to the scene");

        const auto bottomLeft = doublesToVector3d(payload.bottomLeft);
        const auto topRight = doublesToVector3d(payload.topRight);
        Boxf bbox;
        bbox.merge(bottomLeft);
        bbox.merge(topRight);

        const Vector3d s = bbox.getSize() / 2.0;
        const Vector3d c = bbox.getCenter();
        const Vector3d positions[8] = {
            {c.x - s.x, c.y - s.y, c.z - s.z}, {c.x + s.x, c.y - s.y, c.z - s.z}, //    6--------7
            {c.x - s.x, c.y + s.y, c.z - s.z},                                    //   /|       /|
            {c.x + s.x, c.y + s.y, c.z - s.z},                                    //  2--------3 |
            {c.x - s.x, c.y - s.y, c.z + s.z},                                    //  | |      | |
            {c.x + s.x, c.y - s.y, c.z + s.z},                                    //  | 4------|-5
            {c.x - s.x, c.y + s.y, c.z + s.z},                                    //  |/       |/
            {c.x + s.x, c.y + s.y, c.z + s.z}                                     //  0--------1
        };

        const float radius = static_cast<float>(payload.radius);
        for (size_t i = 0; i < 8; ++i)
            model->addSphere(0, {positions[i], radius});

        model->addCylinder(0, {positions[0], positions[1], radius});
        model->addCylinder(0, {positions[2], positions[3], radius});
        model->addCylinder(0, {positions[4], positions[5], radius});
        model->addCylinder(0, {positions[6], positions[7], radius});

        model->addCylinder(0, {positions[0], positions[2], radius});
        model->addCylinder(0, {positions[1], positions[3], radius});
        model->addCylinder(0, {positions[4], positions[6], radius});
        model->addCylinder(0, {positions[5], positions[7], radius});

        model->addCylinder(0, {positions[0], positions[4], radius});
        model->addCylinder(0, {positions[1], positions[5], radius});
        model->addCylinder(0, {positions[2], positions[6], radius});
        model->addCylinder(0, {positions[3], positions[7], radius});

        scene.addModel(std::make_shared<ModelDescriptor>(std::move(model), payload.name));
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_addBox(const AddBoxDetails &details)
{
    Response response;
    try
    {
        if (details.bottomLeft.size() != 3)
            PLUGIN_THROW("Invalid number of double for bottom left corner");
        if (details.topRight.size() != 3)
            PLUGIN_THROW("Invalid number of double for top right corner");
        if (details.color.size() != 3)
            PLUGIN_THROW("Invalid number of double for color");

        auto &scene = _api->getScene();
        auto model = scene.createModel();

        const size_t materialId = 0;
        const auto color = doublesToVector3d(details.color);
        auto material = model->createMaterial(0, "Box");
        material->setDiffuseColor(color);

        const Vector3f minCorner = doublesToVector3d(details.bottomLeft);
        const Vector3f maxCorner = doublesToVector3d(details.topRight);

        TriangleMesh mesh = createBox(minCorner, maxCorner);

        model->getTriangleMeshes()[materialId] = mesh;
        model->markInstancesDirty();

        scene.addModel(std::make_shared<ModelDescriptor>(std::move(model), details.name));
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_addStreamlines(const AddStreamlinesDetails &payload)
{
    Response response;
    try
    {
        const std::string name = payload.name;
        auto &scene = _api->getScene();
        auto model = scene.createModel();

        const auto nbIndices = payload.indices.size();
        const auto nbVertices = payload.vertices.size() / 4;
        const auto nbColors = payload.colors.size() / 4;

        if (nbColors != 0 && nbVertices != nbColors)
            PLUGIN_THROW("Invalid number of colors");

        const bool coloredByOrientation = (nbColors == 0);

        // Create material
        const auto materialId = 0;
        auto material = model->createMaterial(0, "Streamlines");
        material->setDiffuseColor({1, 1, 1});

        for (uint64_t index = 0; index < nbIndices; ++index)
        {
            // Create streamline geometry
            const auto begin = payload.indices[index];
            const auto end = (index == nbIndices - 1 ? nbVertices : payload.indices[index + 1]) - 1;

            if (end - begin < 2)
                continue;

            Vector3fs points;
            Vector3f previousPoint;
            floats radii;
            Vector4fs colors;
            for (uint64_t p = begin; p < end; ++p)
            {
                const auto i = p * 4;
                const Vector3f point = {payload.vertices[i], payload.vertices[i + 1], payload.vertices[i + 2]};
                points.push_back(point);
                radii.push_back(payload.vertices[i + 3]);

                if (coloredByOrientation)
                {
                    if (p == 0)
                        colors.push_back({0.0, 0.0, 0.0, 1.0});
                    else
                    {
                        const Vector3f orientation = 0.5 + 0.5 * normalize(Vector3d(point) - Vector3d(previousPoint));
                        colors.push_back({orientation.x, orientation.y, orientation.z, 1.0});
                    }
                }
                else
                    colors.push_back(
                        {payload.colors[i], payload.colors[i + 1], payload.colors[i + 2], payload.colors[i + 3]});
                previousPoint = point;
            }

            const Streamline streamline(points, colors, radii);
            model->addStreamline(materialId, streamline);
        }

        auto modelDescriptor = std::make_shared<core::ModelDescriptor>(std::move(model), name);
        scene.addModel(modelDescriptor);

        PLUGIN_INFO(1, nbIndices << " streamlines added");
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
        modelIds.ids.push_back(modelId);
    }
    return modelIds;
}

IdsDetails BioExplorerPlugin::_getModelInstances(const ModelIdDetails &payload) const
{
    IdsDetails instanceIds;
    auto &scene = _api->getScene();
    auto modelDescriptor = scene.getModel(payload.modelId);
    if (modelDescriptor)
        for (size_t i = 0; i < std::min(payload.maxNbInstances, modelDescriptor->getInstances().size()); ++i)
            instanceIds.ids.push_back(i);
    else
        instanceIds.ids.push_back(0);
    return instanceIds;
}

NameDetails BioExplorerPlugin::_getModelName(const ModelIdDetails &payload) const
{
    NameDetails modelName;
    auto &scene = _api->getScene();
    auto modelDescriptor = scene.getModel(payload.modelId);
    if (modelDescriptor)
        modelName.name = modelDescriptor->getName();
    return modelName;
}

IdsDetails BioExplorerPlugin::_getMaterialIds(const ModelIdDetails &payload)
{
    IdsDetails materialIds;
    auto &scene = _api->getScene();
    auto modelDescriptor = scene.getModel(payload.modelId);
    if (modelDescriptor)
    {
        for (const auto &material : modelDescriptor->getModel().getMaterials())
            if (material.first != SECONDARY_MODEL_MATERIAL_ID)
                materialIds.ids.push_back(material.first);
    }
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
            PLUGIN_INFO(3, "Modifying materials on model " << modelId);
            auto modelDescriptor = scene.getModel(modelId);
            if (modelDescriptor)
            {
                size_t id = 0;
                for (const auto materialId : payload.materialIds)
                {
                    try
                    {
                        auto material = modelDescriptor->getModel().getMaterial(materialId);
                        if (material)
                        {
                            if (!payload.diffuseColors.empty())
                            {
                                const size_t index = id * 3;
                                material->setDiffuseColor({payload.diffuseColors[index],
                                                           payload.diffuseColors[index + 1],
                                                           payload.diffuseColors[index + 2]});
                                material->setSpecularColor({payload.specularColors[index],
                                                            payload.specularColors[index + 1],
                                                            payload.specularColors[index + 2]});
                            }

                            if (!payload.specularExponents.empty())
                                material->setSpecularExponent(payload.specularExponents[id]);
                            if (!payload.reflectionIndices.empty())
                                material->setReflectionIndex(payload.reflectionIndices[id]);
                            if (!payload.opacities.empty())
                                material->setOpacity(payload.opacities[id]);
                            if (!payload.refractionIndices.empty())
                                material->setRefractionIndex(payload.refractionIndices[id]);
                            if (!payload.emissions.empty())
                                material->setEmission(payload.emissions[id]);
                            if (!payload.glossinesses.empty())
                                material->setGlossiness(payload.glossinesses[id]);
                            if (!payload.shadingModes.empty())
                                material->setShadingMode(static_cast<MaterialShadingMode>(payload.shadingModes[id]));
                            if (!payload.userParameters.empty())
                                material->setUserParameter(payload.userParameters[id]);
                            if (!payload.castUserData.empty())
                                material->setCastUserData(payload.castUserData[id]);
                            if (!payload.clippingModes.empty())
                                material->setClippingMode(static_cast<MaterialClippingMode>(payload.clippingModes[id]));
                            if (!payload.chameleonModes.empty())
                                material->setChameleonMode(
                                    static_cast<MaterialChameleonMode>(payload.chameleonModes[id]));

                            // This is needed to apply modifications.
                            // Changes to the material will be committed
                            // after the rendering of the current frame is
                            // completed. The false parameter is to prevent
                            // the callback to be invoked on every material,
                            // this will be done later on at a scene level
                            material->markModified(false);
                        }
                    }
                    catch (const std::runtime_error &e)
                    {
                        PLUGIN_INFO(1, e.what());
                    }
                    ++id;
                }
            }
            else
                PLUGIN_INFO(3, "Model " << modelId << " is not registered");
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
    const auto &spacing = Vector3d(handler->getSpacing());
    const auto &size = Vector3d(handler->getDimensions()) * spacing;
    const auto &offset = Vector3d(handler->getOffset());
    const Vector3d center{(offset + size / 2.0)};

    const size_t materialId = 0;
    auto material = model->createMaterial(materialId, "default");

    TriangleMesh box = createBox(offset, offset + size);
    model->getTriangleMeshes()[materialId] = box;
    model->updateBounds();
    model->setSimulationHandler(handler);
    setDefaultTransferFunction(*model);

    ModelMetadata metadata;
    metadata["Center"] = std::to_string(center.x) + "," + std::to_string(center.y) + "," + std::to_string(center.z);
    metadata["Size"] = std::to_string(size.x) + "," + std::to_string(size.y) + "," + std::to_string(size.z);
    metadata["Spacing"] = std::to_string(spacing.x) + "," + std::to_string(spacing.y) + "," + std::to_string(spacing.z);
    auto modelDescriptor = std::make_shared<ModelDescriptor>(std::move(model), "Fields", metadata);
    scene.addModel(modelDescriptor);

    size_t modelId = modelDescriptor->getModelID();
    PLUGIN_INFO(3, "Fields model " << modelId << " was successfully created");
    return modelId;
}

Response BioExplorerPlugin::_buildFields(const BuildFieldsDetails &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO(3, "Building Fields from scene");
        auto &scene = _api->getScene();
        auto modelDescriptors = scene.getModelDescriptors();
        for (auto &modelDescriptor : modelDescriptors)
            if (modelDescriptor->getName() == "Fields")
                PLUGIN_THROW("BioExplorer can only handle one single fields model");

        FieldsHandlerPtr handler = std::make_shared<FieldsHandler>(scene, payload.voxelSize, payload.density);
        const auto modelId = _attachFieldsHandler(handler);
        response.contents = std::to_string(modelId);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_importFieldsFromFile(const FileAccessDetails &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO(3, "Importing Fields from " << payload.filename);
        FieldsHandlerPtr handler = std::make_shared<FieldsHandler>(payload.filename);
        _attachFieldsHandler(handler);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_exportFieldsToFile(const ModelIdFileAccessDetails &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO(3, "Exporting fields to " << payload.filename);
        const auto &scene = _api->getScene();
        auto modelDetails = scene.getModel(payload.modelId);
        if (modelDetails)
        {
            auto handler = modelDetails->getModel().getSimulationHandler();
            if (handler)
            {
                FieldsHandler *fieldsHandler = dynamic_cast<FieldsHandler *>(handler.get());
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

Response BioExplorerPlugin::_buildPointCloud(const BuildPointCloudDetails &payload)
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
        material->setDiffuseColor({1.0, 1.0, 1.0});

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
                            tf.getTranslation() + tf.getRotation() * (Vector3d(sphere.center) - tf.getRotationCenter());

                        const Vector3d c = center;
                        if (isClipped(c, clipPlanes))
                            continue;

                        model->addSphere(materialId, {c, static_cast<float>(payload.radius) * sphere.radius});
                    }
                }
            }
        }

        auto md = std::make_shared<ModelDescriptor>(std::move(model), "Point cloud");
        scene.addModel(md);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_setModelsVisibility(const ModelsVisibilityDetails &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO(3, "Setting all models visibility to " << (payload.visible ? "On" : "Off"));
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
        s << "description=" << sceneConfiguration.description << "|scene_size=" << sceneConfiguration.sceneSize.x << ","
          << sceneConfiguration.sceneSize.y << "," << sceneConfiguration.sceneSize.z
          << "|brick_size=" << sceneConfiguration.brickSize.x << "," << sceneConfiguration.brickSize.y << ","
          << sceneConfiguration.brickSize.z << "|visible_bricks=" << _oocManager->getVisibleBricks()
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

ProteinInspectionDetails BioExplorerPlugin::_inspectProtein(const InspectionDetails &details) const
{
    ProteinInspectionDetails result;

    result.hit = false;
    const auto origin = doublesToVector3d(details.origin);
    const auto direction = doublesToVector3d(details.direction);
    double dist = std::numeric_limits<double>::max();
    for (const auto &assembly : _assemblies)
    {
        try
        {
            double t;
            const auto r = assembly.second->inspect(origin, direction, t);
            if (t < dist)
            {
                result = r;
                dist = t;
                result.hit = true;
            }
        }
        catch (const std::runtime_error &e)
        {
            // No hit. Ignore assembly
        }
    }
    return result;
}

Response BioExplorerPlugin::_exportBrickToDatabase(const DatabaseAccessDetails &payload)
{
    Response response;
    try
    {
        const Boxd bounds = vector_to_bounds(payload.lowBounds, payload.highBounds);
        auto &scene = _api->getScene();
        CacheLoader loader(scene);
        loader.exportBrickToDB(payload.brickId, bounds);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

Response BioExplorerPlugin::_addVasculature(const VasculatureDetails &payload)
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addVasculature(payload));
}

Response BioExplorerPlugin::_getVasculatureInfo(const NameDetails &payload) const
{
    ASSEMBLY_CALL(payload.name, getVasculatureInfo());
}

Response BioExplorerPlugin::_setVasculatureReport(const VasculatureReportDetails &payload)
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, setVasculatureReport(payload));
}

Response BioExplorerPlugin::_setVasculatureRadiusReport(const VasculatureRadiusReportDetails &payload)
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, setVasculatureRadiusReport(payload));
}

Response BioExplorerPlugin::_addAstrocytes(const AstrocytesDetails &payload)
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addAstrocytes(payload));
}

Response BioExplorerPlugin::_addNeurons(const NeuronsDetails &payload)
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addNeurons(payload));
}

Response BioExplorerPlugin::_addAtlas(const AtlasDetails &payload)
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addAtlas(payload));
}

Response BioExplorerPlugin::_addWhiteMatter(const WhiteMatterDetails &payload)
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addWhiteMatter(payload));
}

Response BioExplorerPlugin::_addSynapses(const SynapsesDetails &payload)
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addSynapses(payload));
}

Response BioExplorerPlugin::_addSynapseEfficacy(const SynapseEfficacyDetails &payload)
{
    ASSEMBLY_CALL_VOID(payload.assemblyName, addSynapseEfficacy(payload));
}

LookAtResponseDetails BioExplorerPlugin::_lookAt(const LookAtDetails &payload)
{
    LookAtResponseDetails response;
    const auto source = doublesToVector3d(payload.source);
    const auto target = doublesToVector3d(payload.target);
    const auto q = safeQuatlookAt(normalize(target - source));
    response.rotation = {q.x, q.y, q.z, q.w};
    return response;
}

NeuronPointsDetails BioExplorerPlugin::_getNeuronSectionPoints(const NeuronIdSectionIdDetails &payload)
{
    NeuronPointsDetails response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
        {
            response.status = true;
            const auto points = (*it).second->getNeuronSectionPoints(payload);
            for (const auto &point : points)
            {
                response.points.push_back(point.x);
                response.points.push_back(point.y);
                response.points.push_back(point.z);
                response.points.push_back(point.w);
            }
        }
        else
        {
            std::stringstream msg;
            msg << "Assembly not found: " << payload.assemblyName;
            PLUGIN_ERROR(msg.str());
            response.status = false;
        }
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        PLUGIN_ERROR(e.what());
    }
    return response;
}

NeuronPointsDetails BioExplorerPlugin::_getNeuronVaricosities(const NeuronIdDetails &payload)
{
    NeuronPointsDetails response;
    try
    {
        auto it = _assemblies.find(payload.assemblyName);
        if (it != _assemblies.end())
        {
            response.status = true;
            const auto points = (*it).second->getNeuronVaricosities(payload);
            for (const auto &point : points)
            {
                response.points.push_back(point.x);
                response.points.push_back(point.y);
                response.points.push_back(point.z);
            }
        }
        else
        {
            std::stringstream msg;
            msg << "Assembly not found: " << payload.assemblyName;
            PLUGIN_ERROR(msg.str());
            response.status = false;
        }
    }
    catch (const std::runtime_error &e)
    {
        response.status = false;
        PLUGIN_ERROR(e.what());
    }
    return response;
}

Response BioExplorerPlugin::_setSpikeReportVisualizationSettings(const SpikeReportVisualizationSettingsDetails &payload)
{
    Response response;
    try
    {
        PLUGIN_INFO(1, "Setting spike report visualization settings to model " << payload.modelId);
        auto &scene = _api->getScene();
        auto modelDescriptor = scene.getModel(payload.modelId);
        if (!modelDescriptor)
            PLUGIN_THROW("Invalid model id");
        auto handler = modelDescriptor->getModel().getSimulationHandler();
        if (!handler)
            PLUGIN_THROW("Model has no simulation handler");
        auto spikeHandler = dynamic_cast<SpikeSimulationHandler *>(handler.get());
        if (!spikeHandler)
            PLUGIN_THROW("Model does not hold a spike report simulation handler");
        spikeHandler->setVisualizationSettings(payload.restVoltage, payload.spikingVoltage, payload.decaySpeed);
    }
    CATCH_STD_EXCEPTION()
    return response;
}

extern "C" ExtensionPlugin *core_plugin_create(int argc, char **argv)
{
    PLUGIN_INFO(1, " _|_|_|    _|            _|_|_|_|                      _|                                        ");
    PLUGIN_INFO(1, " _|    _|        _|_|    _|        _|    _|  _|_|_|    _|    _|_|    _|  _|_|    _|_|    _|  _|_|");
    PLUGIN_INFO(1, " _|_|_|    _|  _|    _|  _|_|_|      _|_|    _|    _|  _|  _|    _|  _|_|      _|_|_|_|  _|_|    ");
    PLUGIN_INFO(1, " _|    _|  _|  _|    _|  _|        _|    _|  _|    _|  _|  _|    _|  _|        _|        _|      ");
    PLUGIN_INFO(1, " _|_|_|    _|    _|_|    _|_|_|_|  _|    _|  _|_|_|    _|    _|_|    _|          _|_|_|  _|      ");
    PLUGIN_INFO(1, "                                             _|                                                  ");
    PLUGIN_INFO(1, "                                             _|                                                  ");
    PLUGIN_INFO(1, "Initializing BioExplorer plug-in (version " << PACKAGE_VERSION_STRING << ")");
#ifdef USE_CGAL
    PLUGIN_INFO(1, "- CGAL module loaded");
#endif
    PLUGIN_INFO(1, "- Postgresql module loaded");

    return new BioExplorerPlugin(argc, argv);
}

} // namespace bioexplorer
