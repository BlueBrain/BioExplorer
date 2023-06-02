/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include "AbstractCircuitLoader.h"
#include "CellGrowthHandler.h"
#include "SpikeSimulationHandler.h"
#include "VoltageSimulationHandler.h"

#include <plugin/neuroscience/common/MorphologyLoader.h>
#include <plugin/neuroscience/common/ParallelModelContainer.h>
#include <plugin/neuroscience/common/Types.h>

#include <common/Logs.h>
#include <common/Utils.h>

#include <platform/core/common/Timer.h>
#include <platform/core/common/scene/ClipPlane.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>

#include <omp.h>

#ifdef USE_ASSIMP
#include <platform/core/io/MeshLoader.h>
#endif

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
using namespace common;

const strings LOADER_KEYWORDS{"BlueConfig", "CircuitConfig"};
const strings LOADER_EXTENSIONS{"BlueConfig", "BlueConfig3", "CircuitConfig", ".json", "circuit", "CircuitConfig_nrn"};
const std::string GID_PATTERN = "{gid}";
const size_t NB_MATERIALS_PER_INSTANCE = 10;

// Mitochondria density per layer
const std::vector<float> MITOCHONDRIA_DENSITY = {0.0459f, 0.0522f, 0.064f, 0.0774f, 0.0575f, 0.0403f};

AbstractCircuitLoader::AbstractCircuitLoader(Scene &scene, const ApplicationParameters &applicationParameters,
                                             PropertyMap &&loaderParams)
    : Loader(scene)
    , _applicationParameters(applicationParameters)
    , _defaults(loaderParams)
{
}

std::vector<std::string> AbstractCircuitLoader::getSupportedExtensions() const
{
    return LOADER_EXTENSIONS;
}

bool AbstractCircuitLoader::isSupported(const std::string &filename, const std::string & /*extension*/) const
{
    const auto ends_with = [](const std::string &value, const std::string &ending) {
        if (ending.size() > value.size())
            return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    };

    for (const auto &name : LOADER_EXTENSIONS)
        if (ends_with(filename, name))
            return true;

    const auto contains = [](const std::string &value, const std::string &keyword) {
        if (value.size() < keyword.size())
            return false;

        const auto lastSlash = value.find_last_of("/");
        std::string compareTo = value;
        if (lastSlash != std::string::npos)
            compareTo = value.substr(lastSlash + 1);
        return compareTo.find(keyword) != std::string::npos;
    };

    for (const auto &keyw : LOADER_KEYWORDS)
        if (contains(filename, keyw))
            return true;

    return false;
}

std::vector<std::string> AbstractCircuitLoader::_getTargetsAsStrings(const std::string &targets) const
{
    std::vector<std::string> result;
    std::string split;
    std::istringstream ss(targets);
    while (std::getline(ss, split, ','))
        result.push_back(split);
    return result;
}

brain::GIDSet AbstractCircuitLoader::_getGids(const PropertyMap &properties, const brion::BlueConfig &blueConfiguration,
                                              const brain::Circuit &circuit, GIDOffsets &targetGIDOffsets) const
{
    const auto circuitTargets = _getTargetsAsStrings(properties.getProperty<std::string>(PROP_TARGETS.name));
    const auto circuitGids = GIDsAsInts(properties.getProperty<std::string>(PROP_GIDS.name));
    const auto circuitDensity = properties.getProperty<double>(PROP_DENSITY.name);
    const auto randomSeed = properties.getProperty<double>(PROP_RANDOM_SEED.name);
    const auto preSynapticNeuron = properties.getProperty<std::string>(PROP_PRESYNAPTIC_NEURON_GID.name);
    const auto postSynapticNeuron = properties.getProperty<std::string>(PROP_POSTSYNAPTIC_NEURON_GID.name);

    // Pre/post synaptic sanity check
    const bool pre = preSynapticNeuron.empty();
    const bool post = postSynapticNeuron.empty();
    bool prePostSynapticUsecase = false;
    if (!pre || !post)
    {
        if (pre || post)
            PLUGIN_THROW("Pre and post synaptic Gids must be specified");
        prePostSynapticUsecase = true;
    }

    brain::GIDSet gids;
    targetGIDOffsets.push_back(0);

    // Pair synapse usecase
    if (!preSynapticNeuron.empty() && !postSynapticNeuron.empty())
    {
        gids.insert(boost::lexical_cast<Gid>(preSynapticNeuron));
        gids.insert(boost::lexical_cast<Gid>(postSynapticNeuron));
        return gids;
    }

    if (!circuitGids.empty())
    {
        for (const auto gid : circuitGids)
            gids.insert(gid);
        return gids;
    }

    // Determine GIDs according to specified targets
    strings localTargets;
    if (circuitTargets.empty())
    {
        const auto defaultTarget = blueConfiguration.getCircuitTarget();
        localTargets.push_back(defaultTarget);
        PLUGIN_INFO("- No target specified. Loading default one: " << defaultTarget);
    }
    else
        localTargets = circuitTargets;

    for (const auto &target : localTargets)
    {
        const auto targetGids = (circuitDensity == 1.f ? (target.empty() ? circuit.getGIDs() : circuit.getGIDs(target))
                                                       : circuit.getRandomGIDs(circuitDensity, target, randomSeed));
        PLUGIN_INFO("- Target " << target << ": " << targetGids.size() << " cells");
        gids.insert(targetGids.begin(), targetGids.end());
        targetGIDOffsets.push_back(gids.size());
    }

    if (gids.empty())
        PLUGIN_THROW("Selection is empty");
    return gids;
}

CompartmentReportPtr AbstractCircuitLoader::_attachSimulationHandler(const PropertyMap &properties,
                                                                     const brion::BlueConfig &blueConfiguration,
                                                                     Model &model, const ReportType &reportType,
                                                                     brain::GIDSet &gids) const
{
    CompartmentReportPtr compartmentReport{nullptr};

    // Load simulation information from compartment report
    const auto report = properties.getProperty<std::string>(PROP_REPORT.name);
    const auto userDataType = stringToEnum<UserDataType>(properties.getProperty<std::string>(PROP_USER_DATA_TYPE.name));

    if (report.empty() && (userDataType == UserDataType::simulation_offset || userDataType == UserDataType::undefined))
        return nullptr;

    const auto synchronousMode = !properties.getProperty<bool>(PROP_SYNCHRONOUS_MODE.name);

    AbstractSimulationHandlerPtr simulationHandler{nullptr};
    switch (reportType)
    {
    case ReportType::voltages_from_file:
    {
        PLUGIN_INFO("Loading simulation data in " << (synchronousMode ? "a" : "") << "synchronous mode");
        const auto &voltageReport = blueConfiguration.getReportSource(report);
        PLUGIN_INFO("Voltage report: " << voltageReport);
        auto handler = std::make_shared<VoltageSimulationHandler>(voltageReport.getPath(), gids, synchronousMode);
        compartmentReport = handler->getReport();

        // Only keep simulated GIDs
        if (!compartmentReport)
            PLUGIN_THROW("No voltage report was found");
        gids = compartmentReport->getGIDs();

        model.setSimulationHandler(handler);

        simulationHandler = handler;
        setSimulationTransferFunction(model.getTransferFunction());
        break;
    }
    case ReportType::spikes:
    {
        const auto &spikeReport = blueConfiguration.getSpikeSource();
        PLUGIN_INFO("Spike report: " << spikeReport);
        auto handler = std::make_shared<SpikeSimulationHandler>(spikeReport.getPath(), gids);
        model.setSimulationHandler(handler);
        simulationHandler = handler;
        setSimulationTransferFunction(model.getTransferFunction());
        break;
    }
    default:
        if (userDataType == UserDataType::distance_to_soma)
        {
            auto handler = std::make_shared<CellGrowthHandler>(100);
            model.setSimulationHandler(handler);
            setSimulationTransferFunction(model.getTransferFunction(), 0.f);
        }
        else
            PLUGIN_THROW("Unknown report type. Simulation ignored");
    }
    return compartmentReport;
}

bool AbstractCircuitLoader::_isClipped(const Vector3f &position) const
{
    auto &clipPlanes = _scene.getClipPlanes();
    if (clipPlanes.empty())
        return false;

    bool visible = true;
    for (auto clipPlane : clipPlanes)
    {
        const auto &plane = clipPlane->getPlane();
        const Vector3f normal = {plane[0], plane[1], plane[2]};
        const float d = plane[3];
        const float distance = dot(normal, position) + d;
        visible &= (distance > 0.f);
    }
    return !visible;
}

void AbstractCircuitLoader::_filterGIDsWithClippingPlanes(brain::GIDSet &gids, Matrix4fs &transformations) const
{
    // Filter our guids according to clipping planes
    Matrix4fs clippedTransformations;
    brain::GIDSet clippedGids;
    uint64_t i = 0;
    for (const auto gid : gids)
    {
        const auto &transformation = transformations[i];
        if (!_isClipped(get_translation(transformation)))
        {
            clippedTransformations.push_back(transformation);
            clippedGids.insert(gid);
        }
        ++i;
    }
    gids = clippedGids;
    transformations = clippedTransformations;
    PLUGIN_INFO("Clipped circuit: " << gids.size() << " cells");
}

void AbstractCircuitLoader::_filterGIDsWithAreasOfInterest(const uint16_t areasOfInterest, brain::GIDSet &gids,
                                                           Matrix4fs &transformations) const
{
    brain::GIDSet rearrangedGids;
    Matrix4fs rearrangedTransformations;
    Boxf aabb;
    for (const auto &transformation : transformations)
        aabb.merge(get_translation(transformation));

    const auto bbMin = aabb.getMin();
    const auto bbMax = aabb.getMax();
    const float bbSize = aabb.getSize().x / areasOfInterest;
    for (int i = 0; i < areasOfInterest; ++i)
    {
        Boxf area;
        area.merge({bbMin.x + i * bbSize, bbMin.y, bbMin.z});
        area.merge({bbMin.x + (i + 1) * bbSize, bbMax.y, bbMax.z});

        uint64_t j = 0;
        for (const auto &gid : gids)
        {
            if (inBox(get_translation(transformations[j]), area))
            {
                rearrangedGids.insert(gid);
                rearrangedTransformations.push_back(transformations[j]);
                break;
            }
            ++j;
        }
    }
    gids = rearrangedGids;
    transformations = rearrangedTransformations;
    PLUGIN_INFO("Areas of interest: " << gids.size() << " cells");
}

ModelDescriptorPtr AbstractCircuitLoader::importCircuit(const std::string &circuitConfiguration,
                                                        const PropertyMap &properties,
                                                        const LoaderProgress &callback) const
{
    const auto colorScheme =
        stringToEnum<CircuitColorScheme>(properties.getProperty<std::string>(PROP_CIRCUIT_COLOR_SCHEME.name));
    const auto morphologyQuality =
        stringToEnum<AssetQuality>(properties.getProperty<std::string>(PROP_ASSET_QUALITY.name));
    const auto meshFolder = properties.getProperty<std::string>(PROP_MESH_FOLDER.name);
    const auto meshFilenamePattern = properties.getProperty<std::string>(PROP_MESH_FILENAME_PATTERN.name);
    const auto reportType = stringToEnum<ReportType>(properties.getProperty<std::string>(PROP_REPORT_TYPE.name));
    const auto userDataType = stringToEnum<UserDataType>(properties.getProperty<std::string>(PROP_USER_DATA_TYPE.name));
    const auto cellClipping = properties.getProperty<bool>(PROP_CELL_CLIPPING.name);
    const auto areasOfInterest = properties.getProperty<int>(PROP_AREAS_OF_INTEREST.name);

    // Model (one for the whole circuit)
    auto model = _scene.createModel();
    if (!model)
        PLUGIN_THROW("Failed to create model");

    // Open Circuit and select GIDs according to specified target
    callback.updateProgress("Open Brion circuit ...", 0);
    const brion::BlueConfig blueConfiguration(circuitConfiguration);
    callback.updateProgress("Open Brain circuit ...", 0);
    const brain::Circuit circuit(blueConfiguration);

    callback.updateProgress("Getting GIDs...", 0);
    GIDOffsets targetGIDOffsets;
    brain::GIDSet allGids = _getGids(properties, blueConfiguration, circuit, targetGIDOffsets);

    callback.updateProgress("Attaching to simulation data...", 0);

    PLUGIN_INFO("- Attaching simulation handler");
    const auto compartmentReport = _attachSimulationHandler(properties, blueConfiguration, *model, reportType, allGids);

    PLUGIN_INFO("- Applying cell transformations");
    Matrix4fs allTransformations;
    for (const auto &transformation : circuit.getTransforms(allGids))
        allTransformations.push_back(transformation);

    PLUGIN_INFO("- Filtering out guids according to clipping planes");
    if (cellClipping)
        _filterGIDsWithClippingPlanes(allGids, allTransformations);

    PLUGIN_INFO("- Filtering out guids according to aeras of interest");
    if (areasOfInterest != 0)
        _filterGIDsWithAreasOfInterest(areasOfInterest, allGids, allTransformations);

    PLUGIN_INFO("- Identifying layer ids");
    callback.updateProgress("Identifying layer ids...", 0);
    const auto layerIds = _populateLayerIds(properties, blueConfiguration,
                                            colorScheme == CircuitColorScheme::by_target ? allGids : brain::GIDSet());

    callback.updateProgress("Identifying electro-physiology types...", 0);
    PLUGIN_INFO("- Identifying electro-physiologycal types");
    const auto &electrophysiologyTypes =
        circuit.getElectrophysiologyTypes(colorScheme == CircuitColorScheme::by_etype ? allGids : brain::GIDSet());
    callback.updateProgress("Getting cell types...", 0);
    PLUGIN_INFO("- Getting cell types");
    size_ts morphologyTypes;
    if (colorScheme == CircuitColorScheme::by_mtype)
        morphologyTypes = circuit.getMorphologyTypes(allGids);

    callback.updateProgress("Importing cells...", 0);
    PLUGIN_INFO("- Importing cells");
    float maxMorphologyLength = 0.f;
    if (meshFolder.empty())
        maxMorphologyLength =
            _importMorphologies(properties, circuit, *model, allGids, allTransformations, targetGIDOffsets,
                                compartmentReport, layerIds, morphologyTypes, electrophysiologyTypes, callback);
    else
    {
        PLUGIN_INFO("- Importing meshes");
        _importMeshes(properties, *model, allGids, allTransformations, targetGIDOffsets, layerIds, morphologyTypes,
                      electrophysiologyTypes, callback);

        if (compartmentReport != nullptr)
            // If meshes are loaded, and simulation is enabled, a secondary
            // model is created to store the simulation data in the 3D scene
            maxMorphologyLength = _importMorphologies(properties, circuit, *model, allGids, allTransformations,
                                                      targetGIDOffsets, compartmentReport, layerIds, morphologyTypes,
                                                      electrophysiologyTypes, callback, SECONDARY_MODEL_MATERIAL_ID);
    }

    if (userDataType == UserDataType::distance_to_soma)
    {
        PLUGIN_INFO("- Updating cell growth information");
        model->getTransferFunction().setValuesRange({0.f, maxMorphologyLength});
        const auto frameSize = uint64_t(maxMorphologyLength) + 1;
        model->getSimulationHandler()->setFrameSize(frameSize);
        model->getSimulationHandler()->setNbFrames(frameSize);
    }

    PLUGIN_INFO("- Creating custom materials");
    const bool castUserData =
        (compartmentReport || userDataType == UserDataType::distance_to_soma || reportType == ReportType::spikes);
    MorphologyLoader::createMissingMaterials(*model, castUserData);

    PLUGIN_INFO("- Applying default colormap");
    _setDefaultCircuitColorMap(*model);

    PLUGIN_INFO("- Computing circuit center according to soma positions");
    callback.updateProgress("Computing circuit center...", 1);
    Boxf circuitCenter;
    for (const auto &transformation : allTransformations)
        circuitCenter.merge(get_translation(transformation));

    PLUGIN_INFO("- Creating final 3D model");
    ModelMetadata metadata = {{"Report", properties.getProperty<std::string>(PROP_REPORT.name)},
                              {"Report type", properties.getProperty<std::string>(PROP_REPORT_TYPE.name)},
                              {"Targets", properties.getProperty<std::string>(PROP_TARGETS.name)},
                              {"GIDs", properties.getProperty<std::string>(PROP_GIDS.name)},
                              {"Color scheme", enumToString<CircuitColorScheme>(colorScheme)},
                              {"Use simulation model", enumToString<bool>(!meshFolder.empty())},
                              {"Mesh filename pattern", meshFilenamePattern},
                              {"Mesh folder", meshFolder},
                              {"Morphology quality", enumToString<AssetQuality>(morphologyQuality)},
                              {"Number of neurons", std::to_string(allGids.size())},
                              {"Density", std::to_string(properties.getProperty<double>(PROP_DENSITY.name))},
                              {"RandomSeed", std::to_string(properties.getProperty<double>(PROP_RANDOM_SEED.name))},
                              {"CircuitPath", circuitConfiguration}};

    ModelDescriptorPtr modelDescriptor;
    Transformation transformation;
    transformation.setRotationCenter(circuitCenter.getCenter());
    modelDescriptor = std::make_shared<ModelDescriptor>(std::move(model), "Circuit", circuitConfiguration, metadata);
    modelDescriptor->setTransformation(transformation);
    return modelDescriptor;
}

size_t AbstractCircuitLoader::_getMaterialFromCircuitAttributes(const PropertyMap &properties, const uint64_t index,
                                                                const size_t material,
                                                                const GIDOffsets &targetGIDOffsets,
                                                                const size_ts &layerIds, const size_ts &morphologyTypes,
                                                                const size_ts &electrophysiologyTypes,
                                                                const bool forSimulationModel) const
{
    if (material != NO_MATERIAL)
        return material;

    if (forSimulationModel)
        return 0;

    const auto colorScheme =
        stringToEnum<CircuitColorScheme>(properties.getProperty<std::string>(PROP_CIRCUIT_COLOR_SCHEME.name));

    size_t materialId;
    switch (colorScheme)
    {
    case CircuitColorScheme::by_id:
        materialId = NB_MATERIALS_PER_INSTANCE * index;
        break;
    case CircuitColorScheme::by_target:
        for (size_t i = 0; i < targetGIDOffsets.size() - 1; ++i)
            if (index >= targetGIDOffsets[i] && index < targetGIDOffsets[i + 1])
            {
                materialId = NB_MATERIALS_PER_INSTANCE * i;
                break;
            }
        break;
    case CircuitColorScheme::by_etype:
        if (index < electrophysiologyTypes.size())
            materialId = NB_MATERIALS_PER_INSTANCE * electrophysiologyTypes[index];
        else
            PLUGIN_DEBUG("Failed to get neuron E-type")
        break;
    case CircuitColorScheme::by_mtype:
        if (index < morphologyTypes.size())
            materialId = NB_MATERIALS_PER_INSTANCE * morphologyTypes[index];
        else
            PLUGIN_DEBUG("Failed to get neuron M-type")
        break;
    case CircuitColorScheme::by_layer:
        if (index < layerIds.size())
            materialId = NB_MATERIALS_PER_INSTANCE * layerIds[index];
        else
            PLUGIN_DEBUG("Failed to get neuron layer");
        break;
    default:
        materialId = 0;
    }
    return materialId;
}

size_ts AbstractCircuitLoader::_populateLayerIds(const PropertyMap &properties, const brion::BlueConfig &blueConfig,
                                                 const brain::GIDSet &gids) const
{
    size_ts layerIds;
    const auto circuitSource = blueConfig.getCircuitSource();
    try
    {
        PLUGIN_INFO("  - Opening Brion circuit from " << circuitSource);
        const brion::Circuit circuit(circuitSource);
        for (const auto &values : circuit.get(gids, brion::NEURON_LAYER))
            layerIds.push_back(values.size() > 0 ? std::stoi(values[0]) : 0);
    }
    catch (...)
    {
        PLUGIN_INFO("  - Failed to open Brion circuit from " << circuitSource);
        const auto colorScheme =
            stringToEnum<CircuitColorScheme>(properties.getProperty<std::string>(PROP_CIRCUIT_COLOR_SCHEME.name));
        if (colorScheme == CircuitColorScheme::by_layer)
        {
            PLUGIN_ERROR(
                "  - Only MVD2 format is currently supported by Brion "
                "circuits. Color scheme by layer not available for this "
                "circuit");
        }
        else
            PLUGIN_WARN("  - No layer Id could be identified");
    }
    return layerIds;
}

std::string AbstractCircuitLoader::_getMeshFilenameFromGID(const PropertyMap &properties, const uint64_t gid) const
{
    const auto meshFolder = properties.getProperty<std::string>(PROP_MESH_FOLDER.name);
    const auto meshFilenamePattern = properties.getProperty<std::string>(PROP_MESH_FILENAME_PATTERN.name);
    const std::string gidAsString = std::to_string(gid);
    auto tmp = meshFilenamePattern;
    if (!tmp.empty())
        tmp.replace(tmp.find(GID_PATTERN), GID_PATTERN.length(), gidAsString);
    else
        tmp = gidAsString;
    return meshFolder + "/" + tmp;
}

#ifdef USE_ASSIMP
void AbstractCircuitLoader::_importMeshes(const PropertyMap &properties, Model &model, const brain::GIDSet &gids,
                                          const Matrix4fs &transformations, const GIDOffsets &targetGIDOffsets,
                                          const size_ts &layerIds, const size_ts &morphologyTypes,
                                          const size_ts &electrophysiologyTypes, const LoaderProgress &callback) const
{
    MeshLoader meshLoader(_scene);
    const auto morphologyQuality =
        stringToEnum<AssetQuality>(properties.getProperty<std::string>(PROP_ASSET_QUALITY.name));
    const auto meshTransformation = properties.getProperty<bool>(PROP_MESH_TRANSFORMATION.name);

    size_t meshIndex = 0;
    for (const auto &gid : gids)
    {
        const size_t materialId =
            _getMaterialFromCircuitAttributes(properties, meshIndex, NO_MATERIAL, targetGIDOffsets, layerIds,
                                              morphologyTypes, electrophysiologyTypes, false);

        // Load mesh from file
        const auto transformation = meshTransformation ? transformations[meshIndex] : Matrix4f();

        GeometryQuality quality;
        switch (morphologyQuality)
        {
        case AssetQuality::low:
            quality = GeometryQuality::low;
            break;
        case AssetQuality::medium:
            quality = GeometryQuality::medium;
            break;
        default:
            quality = GeometryQuality::high;
            break;
        }
        try
        {
            meshLoader.importMesh(_getMeshFilenameFromGID(properties, gid), LoaderProgress(), model, transformation,
                                  materialId, quality);
        }
        catch (const std::runtime_error &e)
        {
            PLUGIN_WARN(e.what());
        }
        ++meshIndex;
        callback.updateProgress("Loading cells as meshes...", (float)meshIndex / (float)gids.size());
    }
    // Add custom properties to materials
    for (auto &material : model.getMaterials())
    {
        PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_CAST_USER_DATA, true});
        props.setProperty({MATERIAL_PROPERTY_CLIPPING_MODE, static_cast<int>(MaterialClippingMode::no_clipping)});
        material.second->setProperties(props);
    }
}
#else
void AbstractCircuitLoader::_importMeshes(const PropertyMap & /*props*/, Model & /*model*/,
                                          const brain::GIDSet & /*gids*/, const Matrix4fs & /*transformations*/,
                                          const GIDOffsets & /*targetGIDOffsets*/, const size_ts & /*layerIds*/,
                                          const size_ts & /*morphologyTypes*/,
                                          const size_ts & /*electrophysiologyTypes*/,
                                          const LoaderProgress & /*callback*/) const
{
}
#endif

void AbstractCircuitLoader::_setDefaultCircuitColorMap(Model &model) const
{
    const std::vector<Vector3d> colors = {{0.8941176470588236, 0.10196078431372549, 0.10980392156862745},
                                          {0.21568627450980393, 0.49411764705882355, 0.7215686274509804},
                                          {0.30196078431372547, 0.6862745098039216, 0.2901960784313726},
                                          {0.596078431372549, 0.3058823529411765, 0.6392156862745098},
                                          {1.0, 0.4980392156862745, 0.0},
                                          {1.0, 1.0, 0.2},
                                          {0.6509803921568628, 0.33725490196078434, 0.1568627450980392},
                                          {0.9686274509803922, 0.5058823529411764, 0.7490196078431373},
                                          {0.6, 0.6, 0.6},
                                          {0.8941176470588236, 0.10196078431372549, 0.10980392156862745},
                                          {0.21568627450980393, 0.49411764705882355, 0.7215686274509804},
                                          {0.30196078431372547, 0.6862745098039216, 0.2901960784313726},
                                          {0.596078431372549, 0.3058823529411765, 0.6392156862745098},
                                          {1.0, 0.4980392156862745, 0.0},
                                          {1.0, 1.0, 0.2},
                                          {0.6509803921568628, 0.33725490196078434, 0.1568627450980392},
                                          {0.9686274509803922, 0.5058823529411764, 0.7490196078431373},
                                          {0.6, 0.6, 0.6},
                                          {0.8941176470588236, 0.10196078431372549, 0.10980392156862745},
                                          {0.21568627450980393, 0.49411764705882355, 0.7215686274509804}};

    uint64_t i = 0;
    auto &materials = model.getMaterials();
    for (auto &material : materials)
    {
        const auto &color = colors[i % colors.size()];
        material.second->setDiffuseColor(color);
        ++i;
    }
}

void AbstractCircuitLoader::setSimulationTransferFunction(TransferFunction &tf, const float finalOpacity)
{
    tf.setControlPoints({{0, 1}, {1, finalOpacity}});
    // curl https://api.colormaps.io/colormap/unipolar
    tf.setColorMap({"unipolar",
                    {{0.0, 0.0, 0.0},
                     {0.00392156862745098, 0.00392156862745098, 0.12941176470588237},
                     {0.00784313725490196, 0.00784313725490196, 0.25882352941176473},
                     {0.011764705882352941, 0.011764705882352941, 0.39215686274509803},
                     {0.01568627450980392, 0.01568627450980392, 0.5215686274509804},
                     {0.0196078431372549, 0.0196078431372549, 0.6549019607843137},
                     {0.03529411764705882, 0.0784313725490196, 0.6862745098039216},
                     {0.047058823529411764, 0.13333333333333333, 0.7215686274509804},
                     {0.058823529411764705, 0.18823529411764706, 0.7568627450980392},
                     {0.07450980392156863, 0.24705882352941178, 0.788235294117647},
                     {0.08627450980392157, 0.30196078431372547, 0.8235294117647058},
                     {0.09803921568627451, 0.3607843137254902, 0.8588235294117647},
                     {0.11372549019607843, 0.41568627450980394, 0.8901960784313725},
                     {0.12549019607843137, 0.47058823529411764, 0.9254901960784314},
                     {0.13725490196078433, 0.5294117647058824, 0.9568627450980393},
                     {0.2196078431372549, 0.4666666666666667, 0.8745098039215686},
                     {0.30196078431372547, 0.403921568627451, 0.796078431372549},
                     {0.3843137254901961, 0.3411764705882353, 0.7137254901960784},
                     {0.4823529411764706, 0.28627450980392155, 0.596078431372549},
                     {0.5764705882352941, 0.22745098039215686, 0.47843137254901963},
                     {0.6705882352941176, 0.16862745098039217, 0.36470588235294116},
                     {0.7686274509803922, 0.11372549019607843, 0.24705882352941178},
                     {0.8627450980392157, 0.054901960784313725, 0.13333333333333333},
                     {0.9568627450980393, 0.0, 0.01568627450980392},
                     {0.9568627450980393, 0.0196078431372549, 0.01568627450980392},
                     {0.9529411764705882, 0.043137254901960784, 0.01568627450980392},
                     {0.9490196078431372, 0.06666666666666667, 0.01568627450980392},
                     {0.9450980392156862, 0.08627450980392157, 0.01568627450980392},
                     {0.9411764705882353, 0.10980392156862745, 0.01568627450980392},
                     {0.9372549019607843, 0.13333333333333333, 0.011764705882352941},
                     {0.9333333333333333, 0.1568627450980392, 0.011764705882352941},
                     {0.9333333333333333, 0.17647058823529413, 0.011764705882352941},
                     {0.9294117647058824, 0.2, 0.011764705882352941},
                     {0.9254901960784314, 0.2235294117647059, 0.011764705882352941},
                     {0.9215686274509803, 0.24705882352941178, 0.011764705882352941},
                     {0.9176470588235294, 0.26666666666666666, 0.00784313725490196},
                     {0.9137254901960784, 0.2901960784313726, 0.00784313725490196},
                     {0.9098039215686274, 0.3137254901960784, 0.00784313725490196},
                     {0.9098039215686274, 0.33725490196078434, 0.00784313725490196},
                     {0.9058823529411765, 0.3568627450980392, 0.00784313725490196},
                     {0.9019607843137255, 0.3803921568627451, 0.00784313725490196},
                     {0.8980392156862745, 0.403921568627451, 0.00392156862745098},
                     {0.8941176470588236, 0.4235294117647059, 0.00392156862745098},
                     {0.8901960784313725, 0.4470588235294118, 0.00392156862745098},
                     {0.8862745098039215, 0.47058823529411764, 0.00392156862745098},
                     {0.8823529411764706, 0.49411764705882355, 0.00392156862745098},
                     {0.8823529411764706, 0.5137254901960784, 0.00392156862745098},
                     {0.8784313725490196, 0.5372549019607843, 0.0},
                     {0.8745098039215686, 0.5607843137254902, 0.0},
                     {0.8705882352941177, 0.5843137254901961, 0.0},
                     {0.8666666666666667, 0.6039215686274509, 0.0},
                     {0.8627450980392157, 0.6274509803921569, 0.0},
                     {0.8588235294117647, 0.6509803921568628, 0.0},
                     {0.8588235294117647, 0.6745098039215687, 0.0},
                     {0.8588235294117647, 0.6823529411764706, 0.01568627450980392},
                     {0.8627450980392157, 0.6901960784313725, 0.03529411764705882},
                     {0.8666666666666667, 0.7019607843137254, 0.050980392156862744},
                     {0.8705882352941177, 0.7098039215686275, 0.07058823529411765},
                     {0.8705882352941177, 0.7176470588235294, 0.08627450980392157},
                     {0.8745098039215686, 0.7294117647058823, 0.10588235294117647},
                     {0.8784313725490196, 0.7372549019607844, 0.12549019607843137},
                     {0.8823529411764706, 0.7450980392156863, 0.1411764705882353},
                     {0.8823529411764706, 0.7568627450980392, 0.1607843137254902},
                     {0.8862745098039215, 0.7647058823529411, 0.17647058823529413},
                     {0.8901960784313725, 0.7764705882352941, 0.19607843137254902},
                     {0.8941176470588236, 0.7843137254901961, 0.21568627450980393},
                     {0.8980392156862745, 0.792156862745098, 0.23137254901960785},
                     {0.8980392156862745, 0.803921568627451, 0.25098039215686274},
                     {0.9019607843137255, 0.8117647058823529, 0.26666666666666666},
                     {0.9058823529411765, 0.8196078431372549, 0.28627450980392155},
                     {0.9098039215686274, 0.8313725490196079, 0.3058823529411765},
                     {0.9098039215686274, 0.8392156862745098, 0.3215686274509804},
                     {0.9137254901960784, 0.8509803921568627, 0.3411764705882353},
                     {0.9176470588235294, 0.8588235294117647, 0.3568627450980392},
                     {0.9215686274509803, 0.8666666666666667, 0.3764705882352941},
                     {0.9215686274509803, 0.8784313725490196, 0.396078431372549},
                     {0.9254901960784314, 0.8862745098039215, 0.4117647058823529},
                     {0.9294117647058824, 0.8941176470588236, 0.43137254901960786},
                     {0.9333333333333333, 0.9058823529411765, 0.4470588235294118},
                     {0.9372549019607843, 0.9137254901960784, 0.4666666666666667},
                     {0.9372549019607843, 0.9254901960784314, 0.48627450980392156},
                     {0.9411764705882353, 0.9333333333333333, 0.5019607843137255},
                     {0.9450980392156862, 0.9411764705882353, 0.5215686274509804},
                     {0.9490196078431372, 0.9529411764705882, 0.5372549019607843},
                     {0.9490196078431372, 0.9607843137254902, 0.5568627450980392},
                     {0.9529411764705882, 0.9686274509803922, 0.5764705882352941},
                     {0.9568627450980393, 0.9803921568627451, 0.592156862745098},
                     {0.9607843137254902, 0.9882352941176471, 0.611764705882353},
                     {0.9647058823529412, 1.0, 0.6274509803921569},
                     {0.9647058823529412, 1.0, 0.6392156862745098},
                     {0.9647058823529412, 1.0, 0.6470588235294118},
                     {0.9647058823529412, 1.0, 0.6588235294117647},
                     {0.9647058823529412, 1.0, 0.6666666666666666},
                     {0.9686274509803922, 1.0, 0.6745098039215687},
                     {0.9686274509803922, 1.0, 0.6862745098039216},
                     {0.9686274509803922, 1.0, 0.6941176470588235},
                     {0.9686274509803922, 1.0, 0.7019607843137254},
                     {0.9725490196078431, 1.0, 0.7137254901960784},
                     {0.9725490196078431, 1.0, 0.7215686274509804},
                     {0.9725490196078431, 1.0, 0.7294117647058823},
                     {0.9725490196078431, 1.0, 0.7411764705882353},
                     {0.9725490196078431, 1.0, 0.7490196078431373},
                     {0.9764705882352941, 1.0, 0.7568627450980392},
                     {0.9764705882352941, 1.0, 0.7686274509803922},
                     {0.9764705882352941, 1.0, 0.7764705882352941},
                     {0.9764705882352941, 1.0, 0.7843137254901961},
                     {0.9803921568627451, 1.0, 0.796078431372549},
                     {0.9803921568627451, 1.0, 0.803921568627451},
                     {0.9803921568627451, 1.0, 0.8117647058823529},
                     {0.9803921568627451, 1.0, 0.8235294117647058},
                     {0.9803921568627451, 1.0, 0.8313725490196079},
                     {0.984313725490196, 1.0, 0.8431372549019608},
                     {0.984313725490196, 1.0, 0.8509803921568627},
                     {0.984313725490196, 1.0, 0.8588235294117647},
                     {0.984313725490196, 1.0, 0.8705882352941177},
                     {0.9882352941176471, 1.0, 0.8784313725490196},
                     {0.9882352941176471, 1.0, 0.8862745098039215},
                     {0.9882352941176471, 1.0, 0.8980392156862745},
                     {0.9882352941176471, 1.0, 0.9058823529411765},
                     {0.9882352941176471, 1.0, 0.9137254901960784},
                     {0.9921568627450981, 1.0, 0.9254901960784314},
                     {0.9921568627450981, 1.0, 0.9333333333333333},
                     {0.9921568627450981, 1.0, 0.9411764705882353},
                     {0.9921568627450981, 1.0, 0.9529411764705882},
                     {0.996078431372549, 1.0, 0.9607843137254902},
                     {0.996078431372549, 1.0, 0.9686274509803922},
                     {0.996078431372549, 1.0, 0.9803921568627451},
                     {1.0, 1.0, 1.0}}});
    tf.setValuesRange({-80, -10});
}

float AbstractCircuitLoader::_importMorphologies(const PropertyMap &properties, const brain::Circuit &circuit,
                                                 Model &model, const brain::GIDSet &gids,
                                                 const Matrix4fs &transformations, const GIDOffsets &targetGIDOffsets,
                                                 CompartmentReportPtr compartmentReport, const size_ts &layerIds,
                                                 const size_ts &morphologyTypes, const size_ts &electrophysiologyTypes,
                                                 const LoaderProgress &callback, const size_t materialId) const
{
    const auto preSynapticNeuron = properties.getProperty<std::string>(PROP_PRESYNAPTIC_NEURON_GID.name);
    const auto postSynapticNeuron = properties.getProperty<std::string>(PROP_POSTSYNAPTIC_NEURON_GID.name);
    const bool prePostSynapticUsecase = (!preSynapticNeuron.empty() && !postSynapticNeuron.empty());
    const bool loadAfferentSynapses = properties.getProperty<bool>(PROP_LOAD_AFFERENT_SYNAPSES.name);
    const bool loadEfferentSynapses = properties.getProperty<bool>(PROP_LOAD_EFFERENT_SYNAPSES.name);
    const auto position = properties.getProperty<std::array<double, 3>>(PROP_POSITION.name);
    const auto rotation = properties.getProperty<std::array<double, 4>>(PROP_ROTATION.name);
    Transformation transformation;
    transformation.setTranslation(Vector3d(position[0], position[1], position[2]));
    transformation.setRotation(Quaterniond(rotation[0], rotation[1], rotation[2], rotation[3]));

    Timer chrono;
    const auto sectionTypes = MorphologyLoader::getSectionTypesFromProperties(_defaults);
    const bool somasOnly = (sectionTypes.size() == 1 && sectionTypes[0] == brain::neuron::SectionType::soma);

    brain::URIs uris;
    if (!somasOnly)
    {
        PLUGIN_INFO("- Getting cell URIs");
        uris = circuit.getMorphologyURIs(gids);
    }

    PropertyMap morphologyProps(properties);

    std::vector<Gid> localGids;
    for (const auto gid : gids)
        localGids.push_back(gid);

    std::vector<ParallelModelContainer> containers;
    uint64_t morphologyId;
#pragma omp parallel for private(morphologyId)
    for (morphologyId = 0; morphologyId < localGids.size(); ++morphologyId)
    {
        const auto uri = somasOnly ? std::string() : std::string(uris[morphologyId].getPath());
        try
        {
            const auto baseMaterialId =
                _getMaterialFromCircuitAttributes(properties, morphologyId, materialId, targetGIDOffsets, layerIds,
                                                  morphologyTypes, electrophysiologyTypes, false);
            MorphologyLoader loader(_scene, std::move(morphologyProps), transformation);
            loader.setBaseMaterialId(baseMaterialId);

            SynapsesInfo synapsesInfo;
            synapsesInfo.prePostSynapticUsecase = prePostSynapticUsecase;
            const auto gid = localGids[morphologyId];
            if (prePostSynapticUsecase)
            {
                synapsesInfo.preGid = boost::lexical_cast<Gid>(preSynapticNeuron);
                synapsesInfo.postGid = boost::lexical_cast<Gid>(postSynapticNeuron);
            }
            if (loadAfferentSynapses)
                synapsesInfo.afferentSynapses =
                    std::unique_ptr<brain::Synapses>(new brain::Synapses(circuit.getAfferentSynapses({gid})));
            if (loadEfferentSynapses)
                synapsesInfo.efferentSynapses =
                    std::unique_ptr<brain::Synapses>(new brain::Synapses(circuit.getEfferentSynapses({gid})));

            const auto layerId = layerIds.empty() ? 0 : layerIds[morphologyId];
            const auto mitochondriaDensity =
                (layerId < MITOCHONDRIA_DENSITY.size() ? MITOCHONDRIA_DENSITY[layerId] : 0.f);
            ParallelModelContainer modelContainer =
                loader.importMorphology(gid, morphologyProps, uri, morphologyId, synapsesInfo,
                                        transformations[morphologyId], compartmentReport, mitochondriaDensity);
#pragma omp critical
            containers.push_back(modelContainer);
        }
        catch (const std::runtime_error &e)
        {
            PLUGIN_ERROR("Failed to load cell " << morphologyId << " (" << uri << "): " << e.what());
        }
        if (omp_get_thread_num() == 0)
        {
            const float progress = (1 + morphologyId) * omp_get_num_threads();
            PLUGIN_PROGRESS("- Loading cells", progress, localGids.size());

            callback.updateProgress("Loading cells...", progress / localGids.size());
        }
    }
    PLUGIN_INFO("");

    float maxDistanceToSoma = 0.f;
    for (size_t i = 0; i < containers.size(); ++i)
    {
        const float progress = 1.f + i;
        PLUGIN_PROGRESS("- Compiling 3D geometry...", progress, containers.size());
        callback.updateProgress("Compiling 3D geometry...", progress / containers.size());
        auto &container = containers[i];
        maxDistanceToSoma = std::max(container.getMorphologyInfo().maxDistanceToSoma, maxDistanceToSoma);
        container.moveGeometryToModel(model);
    }
    PLUGIN_INFO("");

    PLUGIN_TIMER(chrono.elapsed(), "- " << gids.size() << " cells loaded");
    return maxDistanceToSoma;
}

ModelDescriptorPtr AbstractCircuitLoader::importFromBlob(Blob && /*blob*/, const LoaderProgress & /*callback*/,
                                                         const PropertyMap & /*properties*/) const
{
    PLUGIN_THROW("Load circuit from memory not supported");
}

PropertyMap AbstractCircuitLoader::getProperties() const
{
    return _defaults;
}
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
