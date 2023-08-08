/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "OptiXTypes.h"

#include <platform/core/common/Types.h>

#include <optixu/optixpp_namespace.h>

#include <mutex>
#include <unordered_map>

namespace core
{
// Scene
const std::string CONTEXT_SCENE_TOP_OBJECT = "top_object";
const std::string CONTEXT_SCENE_TOP_SHADOWER = "top_shadower";

// Renderer
const std::string CONTEXT_RENDERER_JITTER = "jitter4";
const std::string CONTEXT_RENDERER_FRAME = "frame";
const std::string CONTEXT_RENDERER_RADIANCE_RAY_TYPE = "radianceRayType";
const std::string CONTEXT_RENDERER_SHADOW_RAY_TYPE = "shadowRayType";
const std::string CONTEXT_RENDERER_SCENE_EPSILON = "sceneEpsilon";
const std::string CONTEXT_RENDERER_AMBIENT_LIGHT_COLOR = "ambientLightColor";
const std::string CONTEXT_RENDERER_BACKGROUND_COLOR = "bgColor";
const std::string CONTEXT_RENDERER_SAMPLES_PER_PIXEL = "samples_per_pixel";

// Camera
const std::string CUDA_FUNC_CAMERA_EXCEPTION = "exception";
const std::string CUDA_FUNC_CAMERA_ENVMAP_MISS = "envmap_miss";
const std::string CUDA_FUNC_BOUNDS = "bounds";
const std::string CUDA_FUNC_INTERSECTION = "intersect";
const std::string CUDA_FUNC_ROBUST_INTERSECTION = "robust_intersect";
const std::string CUDA_FUNC_EXCEPTION = "exception";

const std::string CONTEXT_CAMERA_BAD_COLOR = "bad_color";
const std::string CONTEXT_CAMERA_EYE = "eye";
const std::string CONTEXT_CAMERA_ORIENTATION = "orientation";
const std::string CONTEXT_CAMERA_DIR = "dir";
const std::string CONTEXT_CAMERA_U = "U";
const std::string CONTEXT_CAMERA_V = "V";
const std::string CONTEXT_CAMERA_W = "W";
const std::string CONTEXT_CAMERA_APERTURE_RADIUS = "aperture_radius";
const std::string CONTEXT_CAMERA_FOCAL_SCALE = "focal_scale";
const std::string CONTEXT_CAMERA_FOVY = "fovy";
const std::string CONTEXT_CAMERA_ASPECT = "aspect";
const std::string CONTEXT_CAMERA_OFFSET = "offset";

// Perspective
const std::string CUDA_FUNC_PERSPECTIVE_CAMERA = "perspectiveCamera";
const std::string CONTEXT_CAMERA_STEREO = "stereo";
const std::string CONTEXT_CAMERA_IPD = "interpupillaryDistance";
const std::string CONTEXT_CAMERA_IPD_OFFSET = "ipd_offset";

// Orthographic
const std::string CUDA_FUNC_ORTHOGRAPHIC_CAMERA = "orthographicCamera";
const std::string CONTEXT_CAMERA_HEIGHT = "height";

// Clipping planes
const std::string CONTEXT_ENABLE_CLIPPING_PLANES = "enableClippingPlanes";
const std::string CONTEXT_CLIPPING_PLANES = "clippingPlanes";
const std::string CONTEXT_NB_CLIPPING_PLANES = "nbClippingPlanes";

// Lights
const std::string CONTEXT_LIGHTS = "lights";

// Environment
const std::string CONTEXT_USE_ENVIRONMENT_MAP = "use_envmap";

// Geometry
const std::string CONTEXT_SPHERE_SIZE = "sphere_size";
const std::string CONTEXT_CYLINDER_SIZE = "cylinder_size";
const std::string CONTEXT_CONE_SIZE = "cone_size";

// Material
const std::string CONTEXT_MATERIAL_KA = "Ka";
const std::string CONTEXT_MATERIAL_KD = "Kd";
const std::string CONTEXT_MATERIAL_KS = "Ks";
const std::string CONTEXT_MATERIAL_KR = "Kr";
const std::string CONTEXT_MATERIAL_KO = "Ko";
const std::string CONTEXT_MATERIAL_GLOSSINESS = "glossiness";
const std::string CONTEXT_MATERIAL_REFRACTION_INDEX = "refraction_index";
const std::string CONTEXT_MATERIAL_SPECULAR_EXPONENT = "phong_exp";
const std::string CONTEXT_MATERIAL_SHADING_MODE = "shading_mode";
const std::string CONTEXT_MATERIAL_USER_PARAMETER = "user_parameter";
const std::string CONTEXT_MATERIAL_CAST_USER_DATA = "cast_user_data";
const std::string CONTEXT_MATERIAL_CLIPPING_MODE = "clipping_mode";
const std::string CONTEXT_MATERIAL_RADIANCE_LODS = "radianceLODs";

// Frame buffer
const std::string CONTEXT_STAGE_TONE_MAPPER = "TonemapperSimple";
const std::string CONTEXT_STAGE_DENOISER = "DLDenoiser";
const std::string CONTEXT_INPUT_BUFFER = "input_buffer";
const std::string CONTEXT_OUTPUT_BUFFER = "output_buffer";
const std::string CONTEXT_INPUT_ALBEDO_BUFFER = "input_albedo_buffer";
const std::string CONTEXT_INPUT_NORMAL_BUFFER = "input_normal_buffer";
const std::string CONTEXT_TONE_MAPPER_EXPOSURE = "exposure";
const std::string CONTEXT_TONE_MAPPER_GAMMA = "gamma";
const std::string CONTEXT_DENOISE_BLEND = "blend";
const std::string CONTEXT_ACCUMULATION_BUFFER = "accum_buffer";
const std::string CONTEXT_DENOISED_BUFFER = "denoised_buffer";
const std::string CONTEXT_TONEMAPPED_BUFFER = "tonemapped_buffer";
const std::string CONTEXT_FRAME_NUMBER = "frame_number";

// Volume
const std::string CONTEXT_VOLUME_DATA_TYPE = "volumeDataType";
const std::string CONTEXT_VOLUME_DATA_TYPE_SIZE = "volumeDataTypeSize";
const std::string CONTEXT_VOLUME_DIMENSIONS = "volumeDimensions";
const std::string CONTEXT_VOLUME_OFFSET = "volumeOffset";
const std::string CONTEXT_VOLUME_ELEMENT_SPACING = "volumeElementSpacing";
const std::string CONTEXT_VOLUME_TEXTURE_SAMPLER = "volumeSampler";

// Volume parameters
const std::string CONTEXT_VOLUME_GRADIENT_SHADING_ENABLED = "volumeGradientShadingEnabled";
const std::string CONTEXT_VOLUME_ADAPTIVE_MAX_SAMPLING_RATE = "volumeAdaptiveMaxSamplingRate";
const std::string CONTEXT_VOLUME_ADAPTIVE_SAMPLING = "volumeAdaptiveSampling";
const std::string CONTEXT_VOLUME_SINGLE_SHADE = "volumeSingleShade";
const std::string CONTEXT_VOLUME_PRE_INTEGRATION = "volumePreIntegration";
const std::string CONTEXT_VOLUME_SAMPLING_RATE = "volumeSamplingRate";
const std::string CONTEXT_VOLUME_SPECULAR_COLOR = "volumeSpecularColor";
const std::string CONTEXT_VOLUME_CLIPPING_BOX_LOWER = "volumeClippingBoxLower";
const std::string CONTEXT_VOLUME_CLIPPING_BOX_UPPER = "volumeClippingBoxUpper";

// Transfer function
const std::string CONTEXT_TRANSFER_FUNCTION_COLORS = "tfColors";
const std::string CONTEXT_TRANSFER_FUNCTION_OPACITIES = "tfOpacities";
const std::string CONTEXT_TRANSFER_FUNCTION_SIZE = "tfMapSize";
const std::string CONTEXT_TRANSFER_FUNCTION_MINIMUM_VALUE = "tfMinValue";
const std::string CONTEXT_TRANSFER_FUNCTION_RANGE = "tfRange";

// User data
const std::string CONTEXT_USER_DATA = "simulation_data";

enum class OptixGeometryType
{
    sphere,
    cone,
    cylinder,
    triangleMesh
};

struct OptixShaderProgram
{
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

    std::map<std::string, OptiXShaderProgramPtr> _rendererProgram;
    std::map<std::string, OptiXCameraProgramPtr> _cameraProgram;

    std::map<OptixGeometryType, ::optix::Program> _bounds;
    std::map<OptixGeometryType, ::optix::Program> _intersects;

    std::unordered_map<void*, ::optix::TextureSampler> _optixTextureSamplers;
    std::mutex _mutex;
};
} // namespace core
