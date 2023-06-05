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

#pragma once

#include <platform/core/common/PropertyMap.h>
#include <platform/core/common/geometry/SDFGeometry.h>
#include <platform/core/engineapi/Model.h>

#include <brain/brain.h>
#include <brion/brion.h>

#include <unordered_map>

namespace sonataexplorer
{
namespace neuroscience
{
namespace common
{
class ParallelModelContainer;

using Gid = uint32_t;
using GIDOffsets = std::vector<uint64_t>;
using Matrix4fs = std::vector<core::Matrix4f>;

// Empirical amplitude & frequency
const core::Vector3f DISPLACEMENT_PARAMS = {0.05f, 2.0f, 0.f};

/* Returns a mapping from a name to an enum type. */
template <typename EnumT>
std::vector<std::pair<std::string, EnumT>> enumerateMap();

/* Returns all names for given enum type 'EnumT' */
template <typename EnumT>
inline std::vector<std::string> enumerateNames()
{
    std::vector<std::string> v;
    for (const auto& p : enumerateMap<EnumT>())
        v.push_back(p.first);
    return v;
}

/* Convert a string to an enum. */
template <typename EnumT>
inline EnumT stringToEnum(const std::string& v)
{
    for (const auto& p : enumerateMap<EnumT>())
        if (p.first == v)
            return p.second;

    throw std::runtime_error("Could not match enum '" + v + "'");
    return static_cast<EnumT>(0);
}

/* Convert an enum to a string. */
template <typename EnumT>
inline std::string enumToString(const EnumT v)
{
    for (const auto& p : enumerateMap<EnumT>())
        if (p.second == v)
            return p.first;

    throw std::runtime_error("Could not match enum");
    return "Invalid";
}

// SDF structures
struct SDFMorphologyData
{
    std::vector<core::SDFGeometry> geometries;
    std::vector<std::set<size_t>> neighbours;
    std::vector<size_t> materials;
    std::vector<size_t> localToGlobalIdx;
    std::vector<size_t> bifurcationIndices;
    std::unordered_map<size_t, int> geometrySection;
    std::unordered_map<int, std::vector<size_t>> sectionGeometries;
};

// Asset quality
enum class AssetQuality
{
    low = 0,
    medium = 1,
    high = 2
};

template <>
inline std::vector<std::pair<std::string, AssetQuality>> enumerateMap()
{
    return {{"Low", AssetQuality::low}, {"Medium", AssetQuality::medium}, {"High", AssetQuality::high}};
}

/** Asset color scheme */
enum class AssetColorScheme
{
    none = 0,
    by_segment = 1,
    by_section = 2,
    by_graph = 3
};

template <>
inline std::vector<std::pair<std::string, AssetColorScheme>> enumerateMap()
{
    return {{"None", AssetColorScheme::none},
            {"By segment", AssetColorScheme::by_segment},
            {"By section", AssetColorScheme::by_section},
            {"By graph", AssetColorScheme::by_graph}};
}

using CompartmentReportPtr = std::shared_ptr<brion::CompartmentReport>;

const size_t MATERIAL_OFFSET_SOMA = 1;
const size_t MATERIAL_OFFSET_AXON = 2;
const size_t MATERIAL_OFFSET_DENDRITE = 3;
const size_t MATERIAL_OFFSET_APICAL_DENDRITE = 4;
const size_t MATERIAL_OFFSET_AFFERENT_SYNPASE = 5;
const size_t MATERIAL_OFFSET_EFFERENT_SYNPASE = 6;
const size_t MATERIAL_OFFSET_MITOCHONDRION = 7;
const size_t MATERIAL_OFFSET_NUCLEUS = 8;
const size_t MATERIAL_OFFSET_MYELIN_SHEATH = 9;

enum class CircuitColorScheme
{
    none = 0,
    by_id = 1,
    by_type = 2,
    by_layer = 3,
    by_mtype = 4,
    by_etype = 5,
    by_target = 6
};

enum class ReportType
{
    undefined = 0,
    voltages_from_file = 1,
    spikes = 2
};

/** User data type */
enum class UserDataType
{
    undefined = 0,
    simulation_offset = 1,
    distance_to_soma = 2
};

struct MorphologyInfo
{
    core::Vector3d somaPosition;
    core::Boxd bounds;
    float maxDistanceToSoma;
};

// Synapses
struct SynapsesInfo
{
    std::unique_ptr<brain::Synapses> afferentSynapses{nullptr};
    std::unique_ptr<brain::Synapses> efferentSynapses{nullptr};
    bool prePostSynapticUsecase{false};
    Gid preGid;
    Gid postGid;
};

enum class SynapseType
{
    afferent,
    efferent
};

const std::string CIRCUIT_ON_OFF[2] = {"off", "on"};

template <>
inline std::vector<std::pair<std::string, ReportType>> enumerateMap()
{
    return {{"Undefined", ReportType::undefined},
            {"Voltages from file", ReportType::voltages_from_file},
            {"Spikes", ReportType::spikes}};
}

template <>
inline std::vector<std::pair<std::string, UserDataType>> enumerateMap()
{
    return {{"Undefined", UserDataType::undefined},
            {"Simulation offset", UserDataType::simulation_offset},
            {"Distance to soma", UserDataType::distance_to_soma}};
}

template <>
inline std::vector<std::pair<std::string, CircuitColorScheme>> enumerateMap()
{
    return {{"None", CircuitColorScheme::none},         {"By id", CircuitColorScheme::by_id},
            {"By layer", CircuitColorScheme::by_layer}, {"By mtype", CircuitColorScheme::by_mtype},
            {"By etype", CircuitColorScheme::by_etype}, {"By target", CircuitColorScheme::by_target}};
}

template <>
inline std::vector<std::pair<std::string, bool>> enumerateMap()
{
    return {{"On", true}, {"Off", false}};
}

const core::Property PROP_DENSITY = {"001Density", 1.0, {"Density of cells in the circuit in percent"}};
const core::Property PROP_RANDOM_SEED = {"002RandomSeed", 0.0, {"Random seed for target sub-setting"}};
const core::Property PROP_TARGETS = {"010Targets",
                                       std::string(""),
                                       {"Circuit targets [comma separated list of labels]"}};
const core::Property PROP_PRESYNAPTIC_NEURON_GID = {"012PreNeuron", std::string(""), {"Pre-synaptic neuron GID"}};
const core::Property PROP_POSTSYNAPTIC_NEURON_GID = {"013PostNeuron", std::string(""), {"Post-synaptic neuron GID"}};
const core::Property PROP_REPORT{"020Report", std::string(), {"Circuit report"}};
const core::Property PROP_REPORT_TYPE = {"021ReportType",
                                           enumToString(ReportType::undefined),
                                           enumerateNames<ReportType>(),
                                           {"Type of simulation report"}};
const core::Property PROP_USER_DATA_TYPE = {"022UserDataType",
                                              enumToString(UserDataType::undefined),
                                              enumerateNames<UserDataType>(),
                                              {"Type of data attached to morphology segments"}};
const core::Property PROP_SYNCHRONOUS_MODE = {"023SynchronousMode", false, {"Synchronous mode"}};
const core::Property PROP_CIRCUIT_COLOR_SCHEME = {"030CircuitColorScheme",
                                                    enumToString(CircuitColorScheme::none),
                                                    enumerateNames<CircuitColorScheme>(),
                                                    {"Color scheme to be applied to the circuit"}};
const core::Property PROP_MESH_FOLDER = {"040MeshFolder", std::string(), {"Folder constaining meshes"}};
const core::Property PROP_MESH_FILENAME_PATTERN = {"041MeshFilenamePattern",
                                                     std::string("mesh_{gid}.obj"),
                                                     {"File name pattern for meshes"}};
const core::Property PROP_MESH_TRANSFORMATION = {"042MeshTransformation",
                                                   false,
                                                   {"Apply circuit transformation to meshes"}};
const core::Property PROP_SECTION_TYPE_SOMA = {"052SectionTypeSoma", true, {"Soma"}};
const core::Property PROP_SECTION_TYPE_AXON = {"053SectionTypeAxon", true, {"Axon"}};
const core::Property PROP_SECTION_TYPE_DENDRITE = {"054SectionTypeDendrite", true, {"Dendrite"}};
const core::Property PROP_SECTION_TYPE_APICAL_DENDRITE = {"055SectionTypeApicalDendrite", true, {"Apical Dendrite"}};
const core::Property PROP_MORPHOLOGY_MAX_DISTANCE_TO_SOMA = {"091MaxDistanceToSoma",
                                                               std::numeric_limits<double>::max(),
                                                               {"Maximum distance to soma"}};
const core::Property PROP_CELL_CLIPPING = {"100CellClipping",
                                             false,
                                             {"Clip cells according to scene-defined clipping planes"}};
const core::Property PROP_AREAS_OF_INTEREST = {"101AreasOfInterest", 0, {"Loads only one cell per area of interest"}};
const core::Property PROP_LOAD_AFFERENT_SYNAPSES = {"110LoadAfferentSynapses", false, {"Loads afferent synapses"}};
const core::Property PROP_LOAD_EFFERENT_SYNAPSES = {"111LoadEfferentSynapses", false, {"Loads efferent synapses"}};
const core::Property PROP_INTERNALS = {"120Internals", false, {"Generate internals (mitochondria and nucleus)"}};
const core::Property PROP_EXTERNALS = {"121Externals", false, {"Generate externals (myelin steath)"}};
const core::Property PROP_ALIGN_TO_GRID = {"122AlignToGrid", 0.0, {"Size of the grid to align to (disabled if 0)"}};

const core::Property PROP_GIDS = {"011Gids", std::string(""), {"GIDs [comma separated list of GIDs]"}};
const core::Property PROP_RADIUS_MULTIPLIER = {"050RadiusMultiplier",
                                                 double(1.0),
                                                 {"Multiplier applied to morphology radius"}};
const core::Property PROP_RADIUS_CORRECTION = {"051RadiusCorrection",
                                                 double(0.0),
                                                 {"Value overrideing the radius of the morphology"}};
const core::Property PROP_USE_SDF_SOMA = {"060UseSdfSoma", true, {"Use signed distance field for the soma"}};
const core::Property PROP_USE_SDF_BRANCHES = {"061UseSdfBranches",
                                                true,
                                                {"Use signed distance field for the branches (dendrites and axon)"}};
const core::Property PROP_USE_SDF_NUCLEUS = {"062UseSdfNucleus", true, {"Use signed distance field for the nucleus"}};
const core::Property PROP_USE_SDF_MITOCHONDRIA = {"063UseSdfMitochondria",
                                                    true,
                                                    {"Use signed distance field for the mitochondria"}};
const core::Property PROP_USE_SDF_SYNAPSES = {"064UseSdfSynapses",
                                                true,
                                                {"Use signed distance field for the synapses"}};
const core::Property PROP_USE_SDF_MYELIN_STEATH = {"065UseSdfMyelinSteath",
                                                     true,
                                                     {"Use signed distance field for the myelin steath"}};
const core::Property PROP_DAMPEN_BRANCH_THICKNESS_CHANGERATE = {"066DampenBranchThicknessChangerate",
                                                                  true,
                                                                  {"Dampen branch thickness changerate"}};
const core::Property PROP_ASSET_QUALITY = {"090AssetQuality",
                                             enumToString(AssetQuality::high),
                                             enumerateNames<AssetQuality>(),
                                             {"Quality of the asset"}};
const core::Property PROP_ASSET_COLOR_SCHEME = {"080AssetColorScheme",
                                                  enumToString(AssetColorScheme::none),
                                                  enumerateNames<AssetColorScheme>(),
                                                  {"Color scheme to be applied to the asset"}};
const core::Property PROP_POSITION = {"070Position", std::array<double, 3>{0, 0, 0}, {"Model position"}};
const core::Property PROP_ROTATION = {"071Rotation", std::array<double, 4>{0, 0, 0, 1}, {"Model rotation"}};

} // namespace common
} // namespace neuroscience
} // namespace sonataexplorer
