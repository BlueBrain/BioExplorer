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
namespace engine
{
namespace ospray
{
/*
Camera properties
*/
static const char* OSPRAY_CAMERA_PROPERTY_TYPE_FISHEYE = "fisheye";
static const char* OSPRAY_CAMERA_PROPERTY_TYPE_PERSPECTIVE_PARALLAX = "perspectiveParallax";
static const char* OSPRAY_CAMERA_PROPERTY_TYPE_PANORAMIC = "panoramic";

static constexpr bool OSPRAY_DEFAULT_CAMERA_HALF_SPHERE = true;
static constexpr double OSPRAY_DEFAULT_CAMERA_ZERO_PARALLAX_PLANE = 1.0;

static const Property OSPRAY_CAMERA_PROPERTY_HALF_SPHERE = {"half", OSPRAY_DEFAULT_CAMERA_HALF_SPHERE, {"Half sphere"}};
static const Property OSPRAY_CAMERA_PROPERTY_ZERO_PARALLAX_PLANE = {"zeroParallaxPlane",
                                                                    OSPRAY_DEFAULT_CAMERA_ZERO_PARALLAX_PLANE,
                                                                    {"Zero parallax plane"}};

/*
Engine properties
*/
static const char* OSPRAY_ENGINE_PROPERTY_LOAD_BALANCER_DYNAMIC = "dynamicLoadBalancer";

/*
Renderer properties
*/
static const char* RENDERER_PROPERTY_TYPE_SCIVIS = "scivis";

static const char* OSPRAY_RENDERER_PROPERTY_CAMERA = "camera";
static const char* OSPRAY_RENDERER_PROPERTY_WORLD = "world";
static const char* OSPRAY_RENDERER_PROPERTY_VARIANCE_THRESHOLD = "varianceThreshold";
static const char* OSPRAY_RENDERER_PROPERTY_SAMPLES_PER_PIXEL = "spp";
static const char* OSPRAY_RENDERER_PROPERTY_SHUTTER_CLOSE = "shutterClose";
static const char* OSPRAY_RENDERER_PROPERTY_ANAGLYPH_ENABLED = "anaglyphEnabled";
static const char* OSPRAY_RENDERER_PROPERTY_ANAGLYPH_IPD_OFFSET = "anaglyphIpdOffset";

static constexpr double OSPRAY_DEFAULT_RENDERER_VOLUME_SAMPLING_THRESHOLD = 0.001;
static constexpr double OSPRAY_DEFAULT_RENDERER_VOLUME_SPECULAR_EXPONENT = 20.0;
static constexpr double OSPRAY_DEFAULT_RENDERER_VOLUME_ALPHA_CORRECTION = 0.5;
static constexpr bool DEFAULT_RENDERER_ANAGLYPH_ENABLED = false;

static const Property OSPRAY_RENDERER_AMBIENT_OCCLUSION_DISTANCE = {"aoDistance",
                                                                    10000.,
                                                                    {"Ambient occlusion distance"}};
static const Property OSPRAY_RENDERER_AMBIENT_OCCLUSION_SAMPLES = {
    "aoSamples", int32_t(1), int32_t(0), int32_t(128), {"Ambient occlusion samples"}};
static const Property OSPRAY_RENDERER_AMBIENT_OCCLUSION_ENABLED = {"aoTransparencyEnabled",
                                                                   true,
                                                                   {"Ambient occlusion transparency"}};
static const Property OSPRAY_RENDERER_AMBIENT_OCCLUSION_WEIGHT = {"aoWeight", 0., 0., 1., {"Ambient occlusion weight"}};
static const Property OSPRAY_RENDERER_ONE_SIDED_LIGHTING = {"oneSidedLighting", true, {"One-sided lighting"}};
static const Property OSPRAY_RENDERER_SHADOW_ENABLED = {"shadowsEnabled", false, {"Shadows"}};
static const Property OSPRAY_RENDERER_VOLUME_SAMPLING_THRESHOLD = {"volumeSamplingThreshold",
                                                                   OSPRAY_DEFAULT_RENDERER_VOLUME_SAMPLING_THRESHOLD,
                                                                   0.001,
                                                                   1.,
                                                                   {"Threshold under which sampling is ignored"}};
static const Property OSPRAY_RENDERER_VOLUME_SPECULAR_EXPONENT = {
    "volumeSpecularExponent", OSPRAY_DEFAULT_RENDERER_VOLUME_SPECULAR_EXPONENT, 1., 100., {"Volume specular exponent"}};
static const Property OSPRAY_RENDERER_VOLUME_ALPHA_CORRECTION = {
    "volumeAlphaCorrection", OSPRAY_DEFAULT_RENDERER_VOLUME_ALPHA_CORRECTION, 0.001, 1., {"Volume alpha correction"}};

/*
Volume properties
*/
static const char* OSPRAY_VOLUME_PROPERTY_TYPE_BLOCK_BRICKED = "block_bricked_volume";
static const char* OSPRAY_VOLUME_PROPERTY_TYPE_SHARED_STRUCTURED = "shared_structured_volume";
static const char* OSPRAY_VOLUME_PROPERTY_TYPE_OCTREE = "octree_volume";
static const char* OSPRAY_VOLUME_PROPERTY_DIMENSIONS = "dimensions";
static const char* OSPRAY_VOLUME_PROPERTY_GRID_SPACING = "gridSpacing";
static const char* OSPRAY_VOLUME_VOXEL_TYPE = "voxelType";
static const char* OSPRAY_VOLUME_VOXEL_RANGE = "voxelRange";
static const char* OSPRAY_VOLUME_VOXEL_DATA = "voxelData";
static const char* OSPRAY_VOLUME_OCTREE_INDICES = "octreeIndices";
static const char* OSPRAY_VOLUME_OCTREE_VALUES = "octreeValues";

static const char* OSPRAY_VOLUME_GRADIENT_SHADING_ENABLED = "gradientShadingEnabled";
static const char* OSPRAY_VOLUME_GRADIENT_OFFSET = "gradientOffset";
static const char* OSPRAY_VOLUME_ADAPTIVE_MAX_SAMPLING_RATE = "adaptiveMaxSamplingRate";
static const char* OSPRAY_VOLUME_ADAPTIVE_SAMPLING = "adaptiveSampling";
static const char* OSPRAY_VOLUME_SINGLE_SHADE = "singleShade";
static const char* OSPRAY_VOLUME_PRE_INTEGRATION = "preIntegration";
static const char* OSPRAY_VOLUME_SAMPLING_RATE = "samplingRate";
static const char* OSPRAY_VOLUME_SPECULAR_EXPONENT = "specular";
static const char* OSPRAY_VOLUME_CLIPPING_BOX_LOWER = "volumeClippingBoxLower";
static const char* OSPRAY_VOLUME_CLIPPING_BOX_UPPER = "volumeClippingBoxUpper";
static const char* OSPRAY_VOLUME_USER_PARAMETERS = "userParameters";
static const char* OSPRAY_VOLUME_DIMENSIONS = "volumeDimensions";
static const char* OSPRAY_VOLUME_OFFSET = "volumeOffset";
static const char* OSPRAY_VOLUME_SPACING = "volumeSpacing";

/*
Material properties
*/
static const char* OSPRAY_MATERIAL_TEXTURE_2D = "texture2d";
static const char* OSPRAY_MATERIAL_PROPERTY_TEXTURE_TYPE = "type";
static const char* OSPRAY_MATERIAL_PROPERTY_TEXTURE_SIZE = "size";
static const char* OSPRAY_MATERIAL_PROPERTY_TEXTURE_DATA = "data";

/*
Light properties
*/
static const char* OSPRAY_LIGHT_PROPERTY_COLOR = "color";
static const char* OSPRAY_LIGHT_PROPERTY_INTENSITY = "intensity";
static const char* OSPRAY_LIGHT_PROPERTY_IS_VISIBLE = "isVisible";
static const char* OSPRAY_LIGHT_PROPERTY_POSITION = "position";
static const char* OSPRAY_LIGHT_PROPERTY_DIRECTION = "direction";
static const char* OSPRAY_LIGHT_PROPERTY_RADIUS = "radius";

static const char* OSPRAY_LIGHT_PROPERTY_DISTANT = "distant";
static const char* OSPRAY_LIGHT_PROPERTY_DISTANT_ANGULAR_DIAMETER = "angularDiameter";

static const char* OSPRAY_LIGHT_PROPERTY_AMBIENT = "ambient";

static const char* OSPRAY_LIGHT_PROPERTY_POINT = "point";

static const char* OSPRAY_LIGHT_PROPERTY_QUAD = "quad";
static const char* OSPRAY_LIGHT_PROPERTY_QUAD_EDGE1 = "edge1";
static const char* OSPRAY_LIGHT_PROPERTY_QUAD_EDGE2 = "edge2";

static const char* OSPRAY_LIGHT_PROPERTY_SPOT = "spot";
static const char* OSPRAY_LIGHT_PROPERTY_SPOT_OPENING_ANGLE = "openingAngle";
static const char* OSPRAY_LIGHT_PROPERTY_SPOT_PENUMBRA_ANGLE = "penumbraAngle";

/*
Geometry properties
*/
static const char* OSPRAY_GEOMETRY_PROPERTY_SPHERES = "spheres";
static const char* OSPRAY_GEOMETRY_PROPERTY_SPHERE_OFFSET_CENTER = "offset_center";
static const char* OSPRAY_GEOMETRY_PROPERTY_SPHERE_OFFSET_RADIUS = "offset_radius";
static const char* OSPRAY_GEOMETRY_PROPERTY_SPHERE_BYTES_PER_SPHERE = "bytes_per_sphere";

static const char* OSPRAY_GEOMETRY_PROPERTY_CYLINDERS = "cylinders";
static const char* OSPRAY_GEOMETRY_PROPERTY_CYLINDER_OFFSET_V0 = "offset_v0";
static const char* OSPRAY_GEOMETRY_PROPERTY_CYLINDER_OFFSET_V1 = "offset_v1";
static const char* OSPRAY_GEOMETRY_PROPERTY_CYLINDER_OFFSET_RADIUS = "offset_radius";
static const char* OSPRAY_GEOMETRY_PROPERTY_CYLINDER_BYTES_PER_CYLINDER = "bytes_per_cylinder";

static const char* OSPRAY_GEOMETRY_PROPERTY_CONES = "cones";

static const char* OSPRAY_GEOMETRY_PROPERTY_SDF = "sdfgeometries";
static const char* OSPRAY_GEOMETRY_PROPERTY_SDF_NEIGHBOURS = "neighbours";
static const char* OSPRAY_GEOMETRY_PROPERTY_SDF_GEOMETRIES = "geometries";
static const char* OSPRAY_GEOMETRY_PROPERTY_SDF_EPSILON = "epsilon";
static const char* OSPRAY_GEOMETRY_PROPERTY_SDF_NB_MARCH_ITERATIONS = "nbMarchIterations";
static const char* OSPRAY_GEOMETRY_PROPERTY_SDF_BLEND_FACTOR = "blendFactor";
static const char* OSPRAY_GEOMETRY_PROPERTY_SDF_BLEND_LERP_FACTOR = "blendLerpFactor";
static const char* OSPRAY_GEOMETRY_PROPERTY_SDF_OMEGA = "omega";
static const char* OSPRAY_GEOMETRY_PROPERTY_SDF_DISTANCE = "distance";

static const char* OSPRAY_GEOMETRY_PROPERTY_TRIANGLE_MESH = "trianglemesh";
static const char* OSPRAY_GEOMETRY_PROPERTY_TRIANGLE_MESH_VERTEX = "position";
static const char* OSPRAY_GEOMETRY_PROPERTY_TRIANGLE_MESH_INDEX = "index";
static const char* OSPRAY_GEOMETRY_PROPERTY_TRIANGLE_MESH_NORMAL = "vertex.normal";
static const char* OSPRAY_GEOMETRY_PROPERTY_TRIANGLE_MESH_COLOR = "vertex.color";
static const char* OSPRAY_GEOMETRY_PROPERTY_TRIANGLE_MESH_TEXTURE_COORDINATES = "vertex.texcoord";
static constexpr int OSPRAY_GEOMETRY_DEFAULT_TRIANGLE_MESH_ALPHA_TYPE = 0;
static const char* OSPRAY_GEOMETRY_PROPERTY_TRIANGLE_MESH_ALPHA_TYPE = "alpha_type";
static constexpr int OSPRAY_GEOMETRY_DEFAULT_TRIANGLE_MESH_ALPHA_COMPONENT = 4;
static const char* OSPRAY_GEOMETRY_PROPERTY_TRIANGLE_MESH_ALPHA_COMPONENT = "alpha_component";

static const char* OSPRAY_GEOMETRY_PROPERTY_STREAMLINES = "streamlines";
static const char* OSPRAY_GEOMETRY_PROPERTY_STREAMLINE_VERTEX = "vertex";
static const char* OSPRAY_GEOMETRY_PROPERTY_STREAMLINE_INDEX = "index";
static const char* OSPRAY_GEOMETRY_PROPERTY_STREAMLINE_COLOR = "vertex.color";
static const char* OSPRAY_GEOMETRY_PROPERTY_STREAMLINE_TYPE_SMOOTH = "smooth";

static const char* OSPRAY_GEOMETRY_PROPERTY_CURVES = "curves";
static const char* OSPRAY_GEOMETRY_PROPERTY_CURVE_VERTEX = "vertex";
static const char* OSPRAY_GEOMETRY_PROPERTY_CURVE_INDEX = "index";
static const char* OSPRAY_GEOMETRY_PROPERTY_CURVE_NORMAL = "vertex.normal";
static const char* OSPRAY_GEOMETRY_PROPERTY_CURVE_TANGENT = "vertex.tangent";
static const char* OSPRAY_GEOMETRY_PROPERTY_CURVE_BASIS = "curveBasis";
static const char* OSPRAY_GEOMETRY_PROPERTY_CURVE_TYPE = "curveType";

static const char* OSPRAY_GEOMETRY_PROPERTY_FIELDS = "fields";
static const char* OSPRAY_GEOMETRY_PROPERTY_FIELD_DIMENSIONS = "dimensions";
static const char* OSPRAY_GEOMETRY_PROPERTY_FIELD_SPACING = "spacing";
static const char* OSPRAY_GEOMETRY_PROPERTY_FIELD_OFFSET = "offset";
static const char* OSPRAY_GEOMETRY_PROPERTY_FIELD_INDICES = "indices";
static const char* OSPRAY_GEOMETRY_PROPERTY_FIELD_VALUES = "values";

/*
Frame buffer properties
*/
static const char* OSPRAY_FRAME_BUFFER_PROPERTY_NAME = "name";

/*
Transfer function properties
*/
static const char* OSPRAY_TRANSFER_FUNCTION_PROPERTY_TYPE_PIECEWISE_LINEAR = "piecewise_linear";
static const char* OSPRAY_TRANSFER_FUNCTION_PROPERTY_COLORS = "colors";
static const char* OSPRAY_TRANSFER_FUNCTION_PROPERTY_OPACITIES = "opacities";
static const char* OSPRAY_TRANSFER_FUNCTION_PROPERTY_VALUE_RANGE = "valueRange";

/*
Model properties
*/
static const char* OSPRAY_MODEL_PROPERTY_DYNAMIC_SCENE = "dynamicScene";
static const char* OSPRAY_MODEL_PROPERTY_COMPACT_MODE = "compactMode";
static const char* OSPRAY_MODEL_PROPERTY_ROBUST_MODE = "robustMode";

} // namespace ospray
} // namespace engine
} // namespace core