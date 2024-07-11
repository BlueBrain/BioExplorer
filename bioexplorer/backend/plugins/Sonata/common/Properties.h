/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <platform/core/common/PropertyMap.h>

namespace sonataexplorer
{
/*
Loader properties
*/
static const char* LOADER_BRICK = "brick";
static const char* LOADER_SYNAPSE_CIRCUIT = "synapse_circuit";
static const char* LOADER_MORPHOLOGY = "morphology";
static const char* LOADER_ADVANCED_CIRCUIT = "advanced_circuit";
static const char* LOADER_MORPHOLOGY_COLLAGE = "morphology_collage";
static const char* LOADER_MESH_CIRCUIT = "mesh_circuit";
static const char* LOADER_PAIR_SYNAPSE = "pair_synapse";
static const char* LOADER_ASTROCYTES = "pair_synapse";

/*
Camera properties
*/
static const char* CAMERA_SPHERE_CLIPPING_PERSPECTIVE = "sphere_clipping_perspective";

/*
Renderer properties
*/
static const char* RENDERER_CELL_GROWTH = "cell_growth";
static const char* RENDERER_PROXIMITY = "proximity_detection";

static constexpr double SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_DISTANCE = 1.0;
static constexpr std::array<double, 3> SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_FAR_COLOR = {{1.0, 0.0, 0.0}};
static constexpr std::array<double, 3> SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_NEAR_COLOR = {{0.0, 1.0, 0.0}};
static constexpr bool SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_DIFFERENT_MATERIAL = false;
static constexpr bool SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_SURFACE_SHADING_ENABLED = true;
static constexpr double SONATA_DEFAULT_RENDERER_CELL_GROWTH_SIMULATION_THRESHOLD = 0.0;
static constexpr bool SONATA_DEFAULT_RENDERER_CELL_GROWTH_USE_TRANSFER_FUNCTION_COLOR = false;

static const ::core::Property SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_DISTANCE = {
    "detectionDistance", SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_DISTANCE, {"Detection distance"}};
static const ::core::Property SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_FAR_COLOR = {
    "detectionFarColor", SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_FAR_COLOR, {"Detection far color"}};
static const ::core::Property SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_NEAR_COLOR = {
    "detectionNearColor", SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_NEAR_COLOR, {"Detection near color"}};
static const ::core::Property SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_DIFFERENT_MATERIAL = {
    "detectionOnDifferentMaterial",
    SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_DIFFERENT_MATERIAL,
    {"Detection on different material"}};
static const ::core::Property SONATA_RENDERER_PROPERTY_PROXIMITY_DETECTION_SURFACE_SHADING_ENABLED = {
    "surfaceShadingEnabled", SONATA_DEFAULT_RENDERER_PROXIMITY_DETECTION_SURFACE_SHADING_ENABLED, {"Surface shading"}};

static const ::core::Property SONATA_RENDERER_PROPERTY_CELL_GROWTH_SIMULATION_THRESHOLD = {
    "simulationThreshold",
    SONATA_DEFAULT_RENDERER_CELL_GROWTH_SIMULATION_THRESHOLD,
    0.0,
    1.0,
    {"Simulation threshold"}};
static const ::core::Property SONATA_RENDERER_PROPERTY_USE_TRANSFER_FUNCTION_COLOR = {
    "useTransferFunctionColor",
    SONATA_DEFAULT_RENDERER_CELL_GROWTH_USE_TRANSFER_FUNCTION_COLOR,
    {"Use transfer function color"}};

} // namespace sonataexplorer
