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

#include "OptiXProperties.h"
#include "OptiXTypes.h"
#include "OptiXUtils.h"

#include <platform/core/common/Properties.h>

#include <optixu/optixpp_namespace.h>

#include <mutex>
#include <unordered_map>

namespace core
{
namespace engine
{
namespace optix
{
// Scene
static const char* CONTEXT_SCENE_TOP_OBJECT = "top_object";
static const char* CONTEXT_SCENE_TOP_SHADOWER = "top_shadower";

// Renderer
static const char* CONTEXT_RENDERER_JITTER = "jitter4";
static const char* CONTEXT_RENDERER_FRAME = "frame";
static const char* CONTEXT_RENDERER_RADIANCE_RAY_TYPE = "radianceRayType";
static const char* CONTEXT_RENDERER_SHADOW_RAY_TYPE = "shadowRayType";
static const char* CONTEXT_RENDERER_SCENE_EPSILON = "sceneEpsilon";
static const char* CONTEXT_RENDERER_AMBIENT_LIGHT_COLOR = "ambientLightColor";
static const char* CONTEXT_RENDERER_BACKGROUND_COLOR = "bgColor";
static const char* CONTEXT_RENDERER_SAMPLES_PER_PIXEL = "samples_per_pixel";

// Camera
static const char* CONTEXT_CAMERA_EYE = "eye";
static const char* CONTEXT_CAMERA_ORIENTATION = "orientation";
static const char* CONTEXT_CAMERA_DIR = "dir";
static const char* CONTEXT_CAMERA_U = "U";
static const char* CONTEXT_CAMERA_V = "V";
static const char* CONTEXT_CAMERA_W = "W";
static const char* CONTEXT_CAMERA_APERTURE_RADIUS = CAMERA_PROPERTY_APERTURE_RADIUS.name.c_str();
static const char* CONTEXT_CAMERA_FOCAL_DISTANCE = CAMERA_PROPERTY_FOCAL_DISTANCE.name.c_str();
static const char* CONTEXT_CAMERA_FIELD_OF_VIEW = CAMERA_PROPERTY_FIELD_OF_VIEW.name.c_str();
static const char* CONTEXT_CAMERA_ASPECT = CAMERA_PROPERTY_ASPECT_RATIO.name.c_str();
static const char* CONTEXT_CAMERA_OFFSET = "offset";

// Exception
static const char* CONTEXT_EXCEPTION_BAD_COLOR = "bad_color";

// Perspective
static const char* CUDA_FUNC_PERSPECTIVE_CAMERA = "perspectiveCamera";
static const char* CONTEXT_CAMERA_STEREO = CAMERA_PROPERTY_STEREO.name.c_str();
static const char* CONTEXT_CAMERA_IPD = CAMERA_PROPERTY_INTERPUPILLARY_DISTANCE.name.c_str();
static const char* CONTEXT_CAMERA_IPD_OFFSET = "ipd_offset";

// Orthographic
static const char* CUDA_FUNC_ORTHOGRAPHIC_CAMERA = "orthographicCamera";
static const char* CONTEXT_CAMERA_HEIGHT = CAMERA_PROPERTY_HEIGHT.name.c_str();

// Clipping planes
static const char* CONTEXT_ENABLE_CLIPPING_PLANES = CAMERA_PROPERTY_ENABLE_CLIPPING_PLANES.name.c_str();
static const char* CONTEXT_CLIPPING_PLANES = "clippingPlanes";
static const char* CONTEXT_NB_CLIPPING_PLANES = "nbClippingPlanes";

// Lights
static const char* CONTEXT_LIGHTS = RENDERER_PROPERTY_LIGHTS;

// Environment
static const char* CONTEXT_USE_ENVIRONMENT_MAP = "use_envmap";

// Geometry
static const char* CONTEXT_SPHERE_SIZE = "sphere_size";
static const char* CONTEXT_CYLINDER_SIZE = "cylinder_size";
static const char* CONTEXT_CONE_SIZE = "cone_size";
static const char* CONTEXT_SDF_GEOMETRY_SIZE = "sdf_geometry_size";
static const char* CONTEXT_VOLUME_SIZE = "volume_size";

// Material
static const char* CONTEXT_MATERIAL_KA = "Ka";
static const char* CONTEXT_MATERIAL_KD = "Kd";
static const char* CONTEXT_MATERIAL_KS = "Ks";
static const char* CONTEXT_MATERIAL_KR = "Kr";
static const char* CONTEXT_MATERIAL_KO = "Ko";
static const char* CONTEXT_MATERIAL_GLOSSINESS = "glossiness";
static const char* CONTEXT_MATERIAL_REFRACTION_INDEX = "refraction_index";
static const char* CONTEXT_MATERIAL_SPECULAR_EXPONENT = "phong_exp";
static const char* CONTEXT_MATERIAL_SHADING_MODE = "shading_mode";
static const char* CONTEXT_MATERIAL_USER_PARAMETER = "user_parameter";
static const char* CONTEXT_MATERIAL_CAST_USER_DATA = "cast_user_data";
static const char* CONTEXT_MATERIAL_CLIPPING_MODE = "clipping_mode";
static const char* CONTEXT_MATERIAL_VALUE_RANGE = "value_range";
static const char* CONTEXT_MATERIAL_RADIANCE_LODS = "radianceLODs";

// Frame buffer
static const char* CONTEXT_STAGE_TONE_MAPPER = "TonemapperSimple";
static const char* CONTEXT_STAGE_DENOISER = "DLDenoiser";
static const char* CONTEXT_INPUT_BUFFER = "input_buffer";
static const char* CONTEXT_OUTPUT_BUFFER = "output_buffer";
static const char* CONTEXT_INPUT_ALBEDO_BUFFER = "input_albedo_buffer";
static const char* CONTEXT_INPUT_NORMAL_BUFFER = "input_normal_buffer";
static const char* CONTEXT_TONE_MAPPER_EXPOSURE = "exposure";
static const char* CONTEXT_TONE_MAPPER_GAMMA = "gamma";
static const char* CONTEXT_DENOISE_BLEND = "blend";
static const char* CONTEXT_ACCUMULATION_BUFFER = "accum_buffer";
static const char* CONTEXT_DENOISED_BUFFER = "denoised_buffer";
static const char* CONTEXT_TONEMAPPED_BUFFER = "tonemapped_buffer";
static const char* CONTEXT_FRAME_NUMBER = "frame_number";

// Volume parameters
static const char* CONTEXT_VOLUME_GRADIENT_SHADING_ENABLED = "volumeGradientShadingEnabled";
static const char* CONTEXT_VOLUME_GRADIENT_OFFSET = "volumeGradientOffset";
static const char* CONTEXT_VOLUME_ADAPTIVE_MAX_SAMPLING_RATE = "volumeAdaptiveMaxSamplingRate";
static const char* CONTEXT_VOLUME_ADAPTIVE_SAMPLING = "volumeAdaptiveSampling";
static const char* CONTEXT_VOLUME_SINGLE_SHADE = "volumeSingleShade";
static const char* CONTEXT_VOLUME_PRE_INTEGRATION = "volumePreIntegration";
static const char* CONTEXT_VOLUME_SAMPLING_RATE = "volumeSamplingRate";
static const char* CONTEXT_VOLUME_SPECULAR_COLOR = "volumeSpecularColor";
static const char* CONTEXT_VOLUME_CLIPPING_BOX_LOWER = "volumeClippingBoxLower";
static const char* CONTEXT_VOLUME_CLIPPING_BOX_UPPER = "volumeClippingBoxUpper";
static const char* CONTEXT_VOLUME_USER_PARAMETERS = "volumeUserParameters";

// Geometry parameters
static const char* CONTEXT_GEOMETRY_SDF_EPSILON = "geometrySdfEpsilon";
static const char* CONTEXT_GEOMETRY_SDF_NB_MARCH_ITERATIONS = "geometrySdfNbMarchIterations";
static const char* CONTEXT_GEOMETRY_SDF_BLEND_FACTOR = "geometrySdfBlendFactor";
static const char* CONTEXT_GEOMETRY_SDF_BLEND_LERP_FACTOR = "geometrySdfBlendLerpFactor";
static const char* CONTEXT_GEOMETRY_SDF_OMEGA = "geometrySdfOmega";
static const char* CONTEXT_GEOMETRY_SDF_DISTANCE = "geometrySdfDistance";

// User data
static const char* CONTEXT_USER_DATA = RENDERER_PROPERTY_USER_DATA;

enum class OptixGeometryType
{
    sphere,
    cone,
    cylinder,
    triangleMesh,
    volume,
    streamline,
    sdfGeometry,
};

struct OptixShaderProgram
{
    ~OptixShaderProgram()
    {
        RT_DESTROY(any_hit);
        RT_DESTROY(closest_hit);
        RT_DESTROY(closest_hit_textured);
        RT_DESTROY(exception_program);
    }

    ::optix::Program any_hit{nullptr};
    ::optix::Program closest_hit{nullptr};
    ::optix::Program closest_hit_textured{nullptr};
    ::optix::Program exception_program{nullptr};
};

using OptiXShaderProgramPtr = std::shared_ptr<OptixShaderProgram>;

class OptiXContext
{
public:
    ~OptiXContext();
    static OptiXContext& get();

    ::optix::Context getOptixContext() { return _optixContext; }
    // Camera
    void addCamera(const std::string& name, OptiXCameraProgramPtr program);
    OptiXCameraProgramPtr getCamera(const std::string& name);
    void setCamera(const std::string& name);

    // Geometry
    ::optix::Geometry createGeometry(const OptixGeometryType type);
    ::optix::GeometryGroup createGeometryGroup(const bool compact);
    ::optix::Group createGroup();
    ::optix::Material createMaterial();

    // Textures
    ::optix::TextureSampler createTextureSampler(Texture2DPtr texture);

    // Others
    void addRenderer(const std::string& name, OptiXShaderProgramPtr program);
    OptiXShaderProgramPtr getRenderer(const std::string& name);

    std::unique_lock<std::mutex> getScopeLock() { return std::unique_lock<std::mutex>(_mutex); }

private:
    OptiXContext();

    void _initialize();
    void _printSystemInformation() const;

    static std::unique_ptr<OptiXContext> _context;

    ::optix::Context _optixContext{nullptr};

    std::map<std::string, OptiXShaderProgramPtr> _rendererPrograms;
    std::map<std::string, OptiXCameraProgramPtr> _cameraPrograms;

    std::map<OptixGeometryType, ::optix::Program> _optixBoundsPrograms;
    std::map<OptixGeometryType, ::optix::Program> _optixIntersectionPrograms;

    std::unordered_map<void*, ::optix::TextureSampler> _optixTextureSamplers;
    std::mutex _mutex;
};
} // namespace optix
} // namespace engine
} // namespace core