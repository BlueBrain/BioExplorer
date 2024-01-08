/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#pragma once

#include "TypesEnums.h"

#include <platform/core/common/PropertyMap.h>
#include <platform/core/common/Types.h>

namespace bioexplorer
{
/*
Loader properties
*/

static const std::string SUPPORTED_EXTENTION_DATABASE = "db";

static constexpr std::array<double, 3> LOADER_DEFAULT_POSITION = {{0.0, 0.0, 0.0}};
static constexpr std::array<double, 4> LOADER_DEFAULT_ROTATION = {{0.0, 0.0, 0.0, 1.0}};
static constexpr std::array<double, 3> LOADER_DEFAULT_SCALE = {{1.0, 1.0, 1.0}};

static const core::Property LOADER_PROPERTY_DATABASE_SQL_FILTER{"01DbSqlFilter", std::string(), {"SQL filter"}};
static const core::Property LOADER_PROPERTY_ALIGN_TO_GRID{"02AlignToGrid", 0.0, 0.0, 1000.0, {"Align to grid"}};
static const core::Property LOADER_PROPERTY_POSITION = {"03Position", LOADER_DEFAULT_POSITION, {"Position"}};
static const core::Property LOADER_PROPERTY_ROTATION = {"04Rotation", LOADER_DEFAULT_ROTATION, {"Rotation"}};
static const core::Property LOADER_PROPERTY_SCALE = {"05Scale", LOADER_DEFAULT_SCALE, {"Scale"}};

// Cache
static const char* LOADER_CACHE = "brick";

// Vasculature
static const char* LOADER_VASCULATURE = "DB vasculature";
static const core::Property LOADER_PROPERTY_VASCULATURE_COLOR_SCHEME = {
    "10VasculatureColorScheme",
    core::enumToString(details::VasculatureColorScheme::none),
    core::enumNames<details::VasculatureColorScheme>(),
    {"Color scheme applied to the vasculature"}};
static const core::Property LOADER_PROPERTY_RADIUS_MULTIPLIER = {"20RadiusMultiplier",
                                                                 double(1.0),
                                                                 {"Multiplier applied to radius"}};
static const core::Property LOADER_PROPERTY_VASCULATURE_REALISM_LEVEL_SECTIONS = {"21VasculatureRealismLevelSections",
                                                                                  false,
                                                                                  {"Realistic sections"}};
static const core::Property LOADER_PROPERTY_VASCULATURE_REALISM_LEVEL_BIFURCATIONS = {
    "22VasculatureRealismLevelBifurcations", false, {"Realistic bifurcations"}};
static const core::Property LOADER_PROPERTY_VASCULATURE_REPRESENTATION = {
    "23VasculatureRepresentation",
    core::enumToString(details::VasculatureRepresentation::segment),
    core::enumNames<details::VasculatureRepresentation>(),
    {"Representation"}};

// Morphologies
static const core::Property LOADER_PROPERTY_MORPHOLOGY_COLOR_SCHEME = {
    "10MorphologyColorScheme",
    core::enumToString(morphology::MorphologyColorScheme::none),
    core::enumNames<morphology::MorphologyColorScheme>(),
    {"Color scheme applied to the morphology"}};
static const core::Property LOADER_PROPERTY_POPULATION_COLOR_SCHEME = {
    "11PopulationColorScheme",
    core::enumToString(morphology::PopulationColorScheme::none),
    core::enumNames<morphology::PopulationColorScheme>(),
    {"Population scheme applied to the morphology"}};
static const core::Property LOADER_PROPERTY_MORPHOLOGY_REPRESENTATION = {
    "12MorphologyRepresentation",
    core::enumToString(morphology::MorphologyRepresentation::segment),
    core::enumNames<morphology::MorphologyRepresentation>(),
    {"Morphology representation"}};
static const core::Property LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_SOMA = {"30MorphologyRealismLevelSoma",
                                                                             false,
                                                                             {"Realistic soma"}};
static const core::Property LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_AXON = {"31MorphologyRealismLevelAxon",
                                                                             false,
                                                                             {"Realistic axon"}};
static const core::Property LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_DENDRITE = {"32MorphologyRealismLevelDendrite",
                                                                                 false,
                                                                                 {"Realistic dendrite"}};
static const core::Property LOADER_PROPERTY_MORPHOLOGY_REALISM_LEVEL_INTERNALS = {
    "33MorphologyRealismLevelInternals", false, {"Realistic internals (Nucleus and mitochondria)"}};
static const core::Property LOADER_PROPERTY_MORPHOLOGY_LOAD_SOMA{"21LoadSoma", true, {"Load soma"}};
static const core::Property LOADER_PROPERTY_MORPHOLOGY_LOAD_AXON{"22LoadAxon", false, {"Load axon"}};
static const core::Property LOADER_PROPERTY_MORPHOLOGY_LOAD_DENDRITES{"23LoadDendrites", false, {"Load dendrites"}};
static const core::Property LOADER_PROPERTY_MORPHOLOGY_GENERATE_INTERNALS{
    "24GenerateInternals", false, {"Load internals (Nucleus and mitochondria)"}};

// Astrocytes
static const char* LOADER_ASTROCYTES = "DB astrocytes";
static const core::Property LOADER_PROPERTY_ASTROCYTES_LOAD_END_FEET{"25LoadEndFoot", false, {"Load end feet"}};
static const core::Property LOADER_PROPERTY_ASTROCYTES_LOAD_MICRO_DOMAINS("26LoadMicroDomains", false,
                                                                          {"Load micro-domains"});
static const core::Property LOADER_PROPERTY_ASTROCYTES_VASCULATURE_SCHEMA{"30VasculatureSchema",
                                                                          std::string(),
                                                                          {"Vasculature database schema"}};
// Neurons
static const char* LOADER_NEURONS = "DB neurons";
static const core::Property LOADER_PROPERTY_NEURONS_LOAD_SPINES{"26LoadSpine", false, {"Load spines"}};
static const core::Property LOADER_PROPERTY_NEURONS_LOAD_AXON{"27LoadAxon", false, {"Load axon"}};
static const core::Property LOADER_PROPERTY_NEURONS_LOAD_APICAL_DENDRITES{"28LoadApicalDendrites",
                                                                          false,
                                                                          {"Load apical dendrites"}};
static const core::Property LOADER_PROPERTY_NEURONS_LOAD_BASAL_DENDRITES{"29LoadBasalDendrites",
                                                                         false,
                                                                         {"Load basal dendrites"}};
static const core::Property LOADER_PROPERTY_NEURONS_GENERATE_EXTERNALS{"30GenerateExternals",
                                                                       false,
                                                                       {"Load externals (Myelin steath)"}};
static const core::Property LOADER_PROPERTY_NEURONS_REALISM_LEVEL_EXTERNALS = {"31NeuronsRealismLevelExternals",
                                                                               false,
                                                                               {"Realistic externals (Myelin steath)"}};
static const core::Property LOADER_PROPERTY_NEURONS_REALISM_LEVEL_SPINE = {"31NeuronsRealismLevelSpine",
                                                                           false,
                                                                           {"Realistic spines"}};
static const core::Property LOADER_PROPERTY_NEURONS_SYNAPSE_TYPE = {
    "32NeuronsSynapseType",
    core::enumToString(morphology::MorphologySynapseType::none),
    core::enumNames<morphology::MorphologySynapseType>(),
    {"Type of synapses"}};

// Atlas
static const char* LOADER_ATLAS = "DB brain atlas";

static const core::Property LOADER_PROPERTY_ATLAS_LOAD_CELLS{"30LoadCells", true, {"Load cells"}};
static const core::Property LOADER_PROPERTY_ATLAS_CELL_RADIUS{"31CellRadius", 1.0, 1.0, 1e3, {"Cell radius"}};
static const core::Property LOADER_PROPERTY_ATLAS_CELL_SQL_FILTER{"32CellSqlFilter",
                                                                  std::string(),
                                                                  {"SQL Cell filter"}};
static const core::Property LOADER_PROPERTY_ATLAS_LOAD_MESHES{"33LoadMeshes", true, {"Load meshes"}};
static const core::Property LOADER_PROPERTY_ATLAS_REGION_SQL_FILTER{"34RegionSqlFilter",
                                                                    std::string(),
                                                                    {"SQL Region filter"}};

// White matter
static const char* LOADER_WHITE_MATTER = "DB White matter";

/*
Renderer properties
*/
static const char* RENDERER_GOLGI_STYLE = "bio_explorer_golgi_style";
static const char* RENDERER_DENSITY = "bio_explorer_density";
static const char* RENDERER_POINT_FIELDS = "point_fields";
static const char* RENDERER_VECTOR_FIELDS = "vector_fields";
static const char* RENDERER_PATH_TRACING = "bio_explorer_path_tracing";
static const char* RENDERER_VOXEL = "bio_explorer_voxel";

static constexpr double BIOEXPLORER_DEFAULT_RENDERER_VOXEL_SIMULATION_THRESHOLD = 0.0;

static constexpr double BIOEXPLORER_DEFAULT_RENDERER_FIELDS_MIN_RAY_STEP = 0.001;
static constexpr int BIOEXPLORER_DEFAULT_RENDERER_FIELDS_NB_RAY_STEPS = 8;
static constexpr int BIOEXPLORER_DEFAULT_RENDERER_FIELDS_NB_RAY_REFINEMENT_STEPS = 8;
static constexpr double BIOEXPLORER_DEFAULT_RENDERER_FIELDS_CUTOFF_DISTANCE = 2000.0;

static constexpr double BIOEXPLORER_DEFAULT_RENDERER_RAY_STEP = 2.0;
static constexpr double BIOEXPLORER_DEFAULT_RENDERER_FAR_PLANE = 1000.0;

static constexpr double BIOEXPLORER_DEFAULT_RENDERER_GOLGI_EXPONENT = 5.0;
static constexpr bool BIOEXPLORER_DEFAULT_RENDERER_GOLGI_INVERSE = false;
static constexpr bool BIOEXPLORER_DEFAULT_RENDERER_FIELDS_SHOW_VECTOR_DIRECTIONS = false;

static const core::Property BIOEXPLORER_RENDERER_PROPERTY_VOXEL_SIMULATION_THRESHOLD = {
    "simulationThreshold", BIOEXPLORER_DEFAULT_RENDERER_VOXEL_SIMULATION_THRESHOLD, 0., 1., {"Simulation threshold"}};
static const core::Property BIOEXPLORER_RENDERER_PROPERTY_FIELDS_MIN_RAY_STEP = {
    "minRayStep", BIOEXPLORER_DEFAULT_RENDERER_FIELDS_MIN_RAY_STEP, 0.001, 1.0, {"Smallest ray step"}};
static const core::Property BIOEXPLORER_RENDERER_PROPERTY_FIELDS_NB_RAY_STEPS = {
    "nbRaySteps", BIOEXPLORER_DEFAULT_RENDERER_FIELDS_NB_RAY_STEPS, 1, 2048, {"Number of ray marching steps"}};
static const core::Property BIOEXPLORER_RENDERER_PROPERTY_FIELDS_NB_RAY_REFINEMENT_STEPS = {
    "nbRayRefinementSteps",
    BIOEXPLORER_DEFAULT_RENDERER_FIELDS_NB_RAY_REFINEMENT_STEPS,
    1,
    1000,
    {"Number of ray marching refinement steps"}};
static const core::Property BIOEXPLORER_RENDERER_PROPERTY_FIELDS_CUTOFF_DISTANCE = {
    "cutoff", BIOEXPLORER_DEFAULT_RENDERER_FIELDS_CUTOFF_DISTANCE, 0.0, 1e5, {"Cutoff distance"}};
static const core::Property BIOEXPLORER_RENDERER_PROPERTY_FIELDS_SHOW_VECTOR_DIRECTIONS = {
    "showVectorDirections", BIOEXPLORER_DEFAULT_RENDERER_FIELDS_SHOW_VECTOR_DIRECTIONS, {"Show vector directions"}};

static const core::Property BIOEXPLORER_RENDERER_PROPERTY_RAY_STEP = {
    "rayStep", BIOEXPLORER_DEFAULT_RENDERER_RAY_STEP, 1.0, 1024.0, {"Ray marching step"}};
static const core::Property BIOEXPLORER_RENDERER_PROPERTY_FAR_PLANE = {
    "farPlane", BIOEXPLORER_DEFAULT_RENDERER_FAR_PLANE, 1.0, 1e6, {"Far plane"}};

static const core::Property BIOEXPLORER_RENDERER_PROPERTY_GOLGI_EXPONENT = {
    "exponent", BIOEXPLORER_DEFAULT_RENDERER_GOLGI_EXPONENT, 0.1, 10.0, {"Exponent"}};
static const core::Property BIOEXPLORER_RENDERER_PROPERTY_GOLGI_INVERSE = {"inverse",
                                                                           BIOEXPLORER_DEFAULT_RENDERER_GOLGI_INVERSE,
                                                                           {"Inverse"}};
} // namespace bioexplorer
