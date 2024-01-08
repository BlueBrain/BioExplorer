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

#include <platform/core/common/PropertyMap.h>

namespace core
{
/*
Camera properties
*/
static const char* CAMERA_PROPERTY_TYPE_PERSPECTIVE = "perspective";
static const char* CAMERA_PROPERTY_TYPE_ORTHOGRAPHIC = "orthographic";
static const char* CAMERA_PROPERTY_TYPE_ANAGLYPH = "anaglyph";

static const char* CAMERA_PROPERTY_POSITION = "pos";
static const char* CAMERA_PROPERTY_DIRECTION = "dir";
static const char* CAMERA_PROPERTY_UP_VECTOR = "up";
static const char* CAMERA_PROPERTY_FOCUS_DISTANCE = "focalDistance";
static const char* CAMERA_PROPERTY_CLIPPING_PLANES = "clipPlanes";
static const char* CAMERA_PROPERTY_BUFFER_TARGET = "buffer_target";
static const char* CAMERA_PROPERTY_ENVIRONMENT_MAP = "environmentMap";

static constexpr double DEFAULT_CAMERA_INTERPUPILLARY_DISTANCE = 0.0635;
static constexpr double DEFAULT_CAMERA_FIELD_OF_VIEW = 60.0;
static constexpr double DEFAULT_CAMERA_ASPECT_RATIO = 1.0;
static constexpr double DEFAULT_CAMERA_HEIGHT = 1.0;
static constexpr double DEFAULT_CAMERA_APERTURE_RADIUS = 0.0;
static constexpr double DEFAULT_CAMERA_FOCAL_DISTANCE = 1.0;
static constexpr double DEFAULT_CAMERA_STEREO = false;
static constexpr double DEFAULT_CAMERA_NEAR_CLIP = 0.0;
static constexpr bool DEFAULT_CAMERA_ENABLE_CLIPPING_PLANES = true;

static const Property CAMERA_PROPERTY_FIELD_OF_VIEW = {"fovy", DEFAULT_CAMERA_FIELD_OF_VIEW, {"Field of view"}};
static const Property CAMERA_PROPERTY_APERTURE_RADIUS = {"apertureRadius",
                                                         DEFAULT_CAMERA_APERTURE_RADIUS,
                                                         {"Aperture radius"}};
static const Property CAMERA_PROPERTY_FOCAL_DISTANCE = {"focalDistance",
                                                        DEFAULT_CAMERA_FOCAL_DISTANCE,
                                                        {"Focal Distance"}};
static const Property CAMERA_PROPERTY_ASPECT_RATIO = {"aspect", DEFAULT_CAMERA_ASPECT_RATIO, {"Aspect ratio"}};
static const Property CAMERA_PROPERTY_HEIGHT = {"height", DEFAULT_CAMERA_HEIGHT, {"Height"}};
static const Property CAMERA_PROPERTY_STEREO = {"stereo", DEFAULT_CAMERA_STEREO, {"Stereoscopy"}};
static const Property CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE = {"interpupillaryDistance",
                                                                 DEFAULT_CAMERA_INTERPUPILLARY_DISTANCE,
                                                                 {"Interpupillary distance"}};
static const Property CAMERA_PROPERTY_NEAR_CLIP = {"nearClip", DEFAULT_CAMERA_NEAR_CLIP, 0., 1e6, {"Near clip"}};
static const Property CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES = {"enableClippingPlanes",
                                                                DEFAULT_CAMERA_ENABLE_CLIPPING_PLANES,
                                                                {"Enable clipping planes"}};

/*
Renderer properties
*/
static const char* RENDERER_PROPERTY_TYPE_BASIC = "basic";
static const char* RENDERER_PROPERTY_TYPE_ADVANCED = "advanced";

static const char* RENDERER_PROPERTY_USER_DATA = "userDataBuffer";
static const char* RENDERER_PROPERTY_SECONDARY_MODEL = "secondaryModel";
static const char* RENDERER_PROPERTY_TRANSFER_FUNCTION = "transferFunction";
static const char* RENDERER_PROPERTY_BACKGROUND_MATERIAL = "bgMaterial";
static const char* RENDERER_PROPERTY_LIGHTS = "lights";
static const char* RENDERER_PROPERTY_RANDOM_NUMBER = "randomNumber";

static constexpr bool DEFAULT_RENDERER_FAST_PREVIEW = false;
static constexpr bool DEFAULT_RENDERER_SHOW_BACKGROUND = true;
static constexpr int DEFAULT_RENDERER_RAY_DEPTH = 3;
static constexpr int DEFAULT_RENDERER_MAX_RAY_DEPTH = 30;
static constexpr double DEFAULT_RENDERER_SHADOW_INTENSITY = 0.0;
static constexpr double DEFAULT_RENDERER_SOFT_SHADOW_STRENGTH = 0.0;
static constexpr int DEFAULT_RENDERER_SHADOW_SAMPLES = 1;
static constexpr double DEFAULT_RENDERER_EPSILON_MULTIPLIER = 1.0;
static constexpr double DEFAULT_RENDERER_FOG_START = 1.0;
static constexpr double DEFAULT_RENDERER_FOG_THICKNESS = 1e6;
static constexpr double DEFAULT_RENDERER_GLOBAL_ILLUMINATION_STRENGTH = 0.0;
static constexpr double DEFAULT_RENDERER_GLOBAL_ILLUMINATION_RAY_LENGTH = 1e6;
static constexpr int DEFAULT_RENDERER_GLOBAL_ILLUMINATION_SAMPLES = 0;
static constexpr bool DEFAULT_RENDERER_MATRIX_FILTER = false;
static constexpr double DEFAULT_RENDERER_ALPHA_CORRECTION = 1.0;
static constexpr double DEFAULT_RENDERER_MAX_DISTANCE_TO_SECONDARY_MODEL = 30.0;
static constexpr double DEFAULT_RENDERER_TIMESTAMP = 0.0;

static const Property RENDERER_PROPERTY_TIMESTAMP = {"timestamp", 0.0, 0.0, 1e6, {"Timestamp"}};
static const Property RENDERER_PROPERTY_SHOW_BACKGROUND = {"showBackground", true, {"Show background"}};
static const Property RENDERER_PROPERTY_MAX_RAY_DEPTH = {
    "maxRayDepth", DEFAULT_RENDERER_RAY_DEPTH, 1, DEFAULT_RENDERER_MAX_RAY_DEPTH, {"Maximum ray depth"}};
static const Property RENDERER_PROPERTY_SHADOW_INTENSITY = {
    "shadowIntensity", DEFAULT_RENDERER_SHADOW_INTENSITY, 0.0, 1.0, {"Shadow intensity"}};
static const Property RENDERER_PROPERTY_SOFT_SHADOW_STRENGTH = {
    "softShadowStrength", DEFAULT_RENDERER_SOFT_SHADOW_STRENGTH, 0.0, 1.0, {"Shadow softness"}};
static const Property RENDERER_PROPERTY_SHADOW_SAMPLES = {
    "shadowSamples", DEFAULT_RENDERER_SHADOW_SAMPLES, 1, 64, {"Shadow samples"}};
static const Property RENDERER_PROPERTY_EPSILON_MULTIPLIER = {
    "epsilonMultiplier", 1.0, 1.0, 1000.0, {"Epsilon multiplier"}};
static const Property RENDERER_PROPERTY_FOG_START = {"fogStart", DEFAULT_RENDERER_FOG_START, 0., 1e6, {"Fog start"}};
static const Property RENDERER_PROPERTY_FOG_THICKNESS = {
    "fogThickness", DEFAULT_RENDERER_FOG_THICKNESS, 0.0, 1e6, {"Fog thickness"}};
static const Property RENDERER_PROPERTY_GLOBAL_ILLUMINATION_STRENGTH = {
    "giStrength", DEFAULT_RENDERER_GLOBAL_ILLUMINATION_STRENGTH, 0.0, 1.0, {"Global illumination strength"}};
static const Property RENDERER_PROPERTY_GLOBAL_ILLUMINATION_RAY_LENGTH = {
    "giRayLength", DEFAULT_RENDERER_GLOBAL_ILLUMINATION_RAY_LENGTH, 0.0, 1e6, {"Global illumination ray length"}};
static const Property RENDERER_PROPERTY_GLOBAL_ILLUMINATION_SAMPLES = {
    "giSamples", DEFAULT_RENDERER_GLOBAL_ILLUMINATION_SAMPLES, 0, 64, {"Global illumination samples"}};
static const Property RENDERER_PROPERTY_MATRIX_FILTER = {"matrixFilter",
                                                         DEFAULT_RENDERER_MATRIX_FILTER,
                                                         {"Feels like being in the Matrix"}};
static const Property RENDERER_PROPERTY_ALPHA_CORRECTION = {
    "alphaCorrection", DEFAULT_RENDERER_ALPHA_CORRECTION, 0.001, 1., {"Alpha correction"}};
static const Property RENDERER_PROPERTY_MAX_DISTANCE_TO_SECONDARY_MODEL = {
    "maxDistanceToSecondaryModel",
    DEFAULT_RENDERER_MAX_DISTANCE_TO_SECONDARY_MODEL,
    0.1,
    100.0,
    {"Maximum distance to secondary model"}};
static const Property RENDERER_PROPERTY_FAST_PREVIEW = {"fastPreview", DEFAULT_RENDERER_FAST_PREVIEW, {"Fast preview"}};

/*
Material properties
*/
static const char* MATERIAL_PROPERTY_OPACITY = "d";
static const char* MATERIAL_PROPERTY_MAP_OPACITY = "map_d";
static const char* MATERIAL_PROPERTY_DIFFUSE_COLOR = "kd";
static const char* MATERIAL_PROPERTY_MAP_DIFFUSE_COLOR = "map_kd";
static const char* MATERIAL_PROPERTY_SPECULAR_COLOR = "ks";
static const char* MATERIAL_PROPERTY_MAP_SPECULAR_COLOR = "map_ks";
static const char* MATERIAL_PROPERTY_SPECULAR_INDEX = "ns";
static const char* MATERIAL_PROPERTY_MAP_SPECULAR_INDEX = "map_ns";
static const char* MATERIAL_PROPERTY_MAP_BUMP = "map_bump";
static const char* MATERIAL_PROPERTY_REFRACTION = "refraction";
static const char* MATERIAL_PROPERTY_MAP_REFRACTION = "map_refraction";
static const char* MATERIAL_PROPERTY_REFLECTION = "kr";
static const char* MATERIAL_PROPERTY_MAP_REFLECTION = "map_kr";
static const char* MATERIAL_PROPERTY_EMISSION = "a";
static const char* MATERIAL_PROPERTY_MAP_EMISSION = "map_a";
static const char* MATERIAL_PROPERTY_SHADING_MODE = "shading_mode";
static const char* MATERIAL_PROPERTY_USER_PARAMETER = "user_parameter";
static const char* MATERIAL_PROPERTY_GLOSSINESS = "glossiness";
static const char* MATERIAL_PROPERTY_CAST_USER_DATA = "cast_user_data";
static const char* MATERIAL_PROPERTY_CLIPPING_MODE = "clipping_mode";
static const char* MATERIAL_PROPERTY_CHAMELEON_MODE = "chameleon_mode";
static const char* MATERIAL_PROPERTY_NODE_ID = "node_id";

/*
Common properties
*/
static const char* DEFAULT = "default";

static constexpr double DEFAULT_COMMON_EXPOSURE = 1.0;
static constexpr bool DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER = false;

static const Property COMMON_PROPERTY_EXPOSURE = {"mainExposure", DEFAULT_COMMON_EXPOSURE, 1., 20., {"Exposure"}};
static const Property COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER = {"useHardwareRandomizer",
                                                                 DEFAULT_COMMON_USE_HARDWARE_RANDOMIZER,
                                                                 {"Use hardware randomizer"}};

} // namespace core
