/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
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
