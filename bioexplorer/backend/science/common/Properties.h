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

namespace bioexplorer
{
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
