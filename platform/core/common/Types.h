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

#include <platform/core/common/MathTypes.h>
#include <platform/core/common/utils/EnumUtils.h>

#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <vector>

using int64 = ::int64_t;
using uint64 = ::uint64_t;
using int32 = ::int32_t;
using uint32 = ::uint32_t;
using int16 = ::int16_t;
using uint16 = ::uint16_t;
using int8 = ::int8_t;
using uint8 = ::uint8_t;
using index_t = ::int64_t;
using strings = std::vector<std::string>;
using floats = std::vector<float>;
using ints = std::vector<int>;
using uints = std::vector<unsigned int>;
using int8_ts = std::vector<int8_t>;
using uint8_ts = std::vector<uint8_t>;
using int16_ts = std::vector<int16_t>;
using uint16_ts = std::vector<uint16_t>;
using int32_ts = std::vector<int32_t>;
using uint32_ts = std::vector<uint32_t>;
using int64_ts = std::vector<int64_t>;
using uint64_ts = std::vector<uint64_t>;
using size_ts = std::vector<size_t>;
using StringMap = std::map<std::string, std::string>;
using Color = core::Vector3d;
using Palette = std::vector<Color>;
using Quaternions = std::vector<core::Quaterniond>;
using bools = std::vector<bool>;
using doubles = std::vector<double>;
using strings = std::vector<std::string>;
using Vector3ds = std::vector<core::Vector3d>;
using Vector3dm = std::map<uint64_t, core::Vector3d>;
using Vector4ds = std::vector<core::Vector4d>;
using Vector2uis = std::vector<core::Vector2ui>;
using Vector3uis = std::vector<core::Vector3ui>;
using uint8_ts = std::vector<uint8_t>;
using uint8_tm = std::map<uint64_t, uint8_t>;
using uint32_ts = std::vector<uint32_t>;
using uint64_ts = std::vector<uint64_t>;
using uint64_tm = std::map<uint64_t, uint64_t>;
using CommandLineArguments = std::map<std::string, std::string>;

namespace core
{
// Forward declarations
class Core;

class ActionInterface;
using ActionInterfacePtr = std::shared_ptr<ActionInterface>;

class Engine;

class Scene;
using ScenePtr = std::shared_ptr<Scene>;

class AbstractManipulator;

class Camera;
using CameraPtr = std::shared_ptr<Camera>;

class TransferFunction;
class Renderer;
using RendererPtr = std::shared_ptr<Renderer>;

class FrameBuffer;
using FrameBufferPtr = std::shared_ptr<FrameBuffer>;

class Model;
using ModelPtr = std::unique_ptr<Model>;
using ModelMetadata = std::map<std::string, std::string>;

class Transformation;
using Transformations = std::vector<Transformation>;

class ModelInstance;
class ModelParams;
class ModelDescriptor;
using ModelDescriptorPtr = std::shared_ptr<ModelDescriptor>;
using ModelDescriptors = std::vector<ModelDescriptorPtr>;
using ModelInstances = std::vector<ModelInstance>;

class Material;
using MaterialPtr = std::shared_ptr<Material>;
using MaterialMap = std::map<size_t, MaterialPtr>;

class ClipPlane;
using ClipPlanePtr = std::shared_ptr<ClipPlane>;
using ClipPlanes = std::vector<ClipPlanePtr>;

struct Sphere;
using Spheres = std::vector<Sphere>;
using SpheresMap = std::map<size_t, Spheres>;

struct Cylinder;
using Cylinders = std::vector<Cylinder>;
using CylindersMap = std::map<size_t, Cylinders>;

struct Cone;
using Cones = std::vector<Cone>;
using ConesMap = std::map<size_t, Cones>;

struct SDFBezier;
using SDFBeziers = std::vector<SDFBezier>;
using SDFBeziersMap = std::map<size_t, SDFBeziers>;

struct TriangleMesh;
using TriangleMeshMap = std::map<size_t, TriangleMesh>;

struct StreamlinesData;
using StreamlinesDataMap = std::map<size_t, StreamlinesData>;

struct Curve;
using Curves = std::vector<Curve>;
using CurvesMap = std::map<size_t, Curves>;

class Volume;
class BrickedVolume;
class SharedDataVolume;
using VolumePtr = std::shared_ptr<Volume>;
using SharedDataVolumePtr = std::shared_ptr<SharedDataVolume>;
using BrickedVolumePtr = std::shared_ptr<BrickedVolume>;
using Volumes = std::vector<VolumePtr>;
using VolumesMap = std::map<size_t, Volumes>;

class Texture2D;
using Texture2DPtr = std::shared_ptr<Texture2D>;
using TexturesMap = std::map<std::string, Texture2DPtr>;

class Light;
using LightPtr = std::shared_ptr<Light>;
using Lights = std::vector<LightPtr>;

class DirectionalLight;
using DirectionalLightPtr = std::shared_ptr<DirectionalLight>;

class SphereLight;
using SphereLightPtr = std::shared_ptr<SphereLight>;

class QuadLight;
using QuadLightPtr = std::shared_ptr<QuadLight>;

class SpotLight;
using SpotLightPtr = std::shared_ptr<SpotLight>;

class AbstractSimulationHandler;
using AbstractSimulationHandlerPtr = std::shared_ptr<AbstractSimulationHandler>;

class AbstractParameters;
class AnimationParameters;
class ApplicationParameters;
class GeometryParameters;
class ParametersManager;
class RenderingParameters;
class VolumeParameters;

class PluginAPI;
class ExtensionPlugin;

class KeyboardHandler;

class MeshLoader;

class Statistics;

/** Supported engines */
static const char* ENGINE_OSPRAY = "ospray";
static const char* ENGINE_OPTIX_6 = "optix6";
static const char* ENGINE_OPTIX_7 = "optix7";

/** Define the frame buffer format */
enum class FrameBufferFormat
{
    rgba_i8,
    bgra_i8,
    rgb_i8,
    rgb_f32,
    none
};

/** Accumulation types */
enum class AccumulationType
{
    linear = 0,
    ai_denoised = 1,
};

/** Geometry quality */
enum class GeometryQuality
{
    low,
    medium,
    high
};

/** Some 'special' materials are used by Core to accomplish specific features
 *  such as bounding boxes.
 */
static const size_t NO_MATERIAL = std::numeric_limits<size_t>::max();
static const size_t BOUNDINGBOX_MATERIAL_ID = NO_MATERIAL - 1;
static const size_t SECONDARY_MODEL_MATERIAL_ID = NO_MATERIAL - 2;
static const size_t VOLUME_MATERIAL_ID = NO_MATERIAL - 3;

static const std::string IRRADIANCE_MAP = "-irradiance";
static const std::string RADIANCE_MAP = "-radiance";
static const std::string BRDF_LUT = "-brdfLUT";

enum class TextureType : uint8_t
{
    diffuse = 0,
    normals,
    bump,
    specular,
    emissive,
    opacity,
    reflection,
    refraction,
    occlusion,
    radiance,
    irradiance,
    brdf_lut,
    volume,
    transfer_function
};

static const strings textureTypeToString{
    "albedoMetallic_map", "normalRoughness_map",  "bump_map",      "aoEmissive_map", "map_ns",         "map_d",
    "map_reflection",     "map_refraction",       "map_occlusion", "radiance_map",   "irradiance_map", "brdf_lut",
    "volume_map",         "transfer_function_map"};

enum class MemoryMode
{
    shared,
    replicated
};

enum class MaterialsColorMap
{
    random,         // Random materials including transparency, reflection,
                    // and light emission
    shades_of_grey, // 255 shades of grey
    gradient,       // Gradient from blue to yellow through green
    pastel          // Random pastel colors
};

/**
 * The different modes for moving the camera.
 */
enum class CameraMode
{
    flying,
    inspect
};

/** A clip plane is defined by a normal and a distance expressed
 * in absolute value of the coordinate system. Values are stored
 * in a Vector4, with the following order: nx, ny, nz and d
 */
using Plane = std::array<double, 4>;
using Planes = std::vector<Plane>;

struct RenderInput
{
    Vector2i windowSize;

    Vector3d position;
    Vector3d target;
    Quaterniond orientation;
};

struct RenderOutput
{
    Vector2i frameSize;
    uint8_ts colorBuffer;
    floats floatBuffer;
    FrameBufferFormat colorBufferFormat;
};

class Progress;

class AbstractTask;
using TaskPtr = std::shared_ptr<AbstractTask>;

struct Blob
{
    std::string type;
    std::string name;
    uint8_ts data;
};

class Loader;
using LoaderPtr = std::unique_ptr<Loader>;

enum class DataType
{
    FLOAT,
    DOUBLE,
    UINT8,
    UINT16,
    UINT32,
    INT8,
    INT16,
    INT32
};

class PropertyMap;
class PropertyObject;

enum class Execution
{
    sync,
    async
};

/** Description for RPC with no parameter. */
struct RpcDescription
{
    std::string methodName;
    std::string methodDescription;
    Execution type{Execution::sync};
};

/** Description for RPC with one parameter. */
struct RpcParameterDescription : RpcDescription
{
    RpcParameterDescription(const std::string& methodName_, const std::string& methodDescription_,
                            const Execution type_, const std::string& paramName_, const std::string& paramDescription_)
        : RpcDescription{methodName_, methodDescription_, type_}
        , paramName(paramName_)
        , paramDescription(paramDescription_)
    {
    }

    RpcParameterDescription(const std::string& methodName_, const std::string& methodDescription_,
                            const std::string& paramName_, const std::string& paramDescription_)
        : RpcDescription{methodName_, methodDescription_, Execution::sync}
        , paramName(paramName_)
        , paramDescription(paramDescription_)
    {
    }

    std::string paramName;
    std::string paramDescription;
};

enum class BVHFlag
{
    dynamic,
    compact,
    robust
};

///////////////////////////////////////////////////////////////////////////

template <>
inline std::vector<std::pair<std::string, GeometryQuality>> enumMap()
{
    return {{"low", GeometryQuality::low}, {"medium", GeometryQuality::medium}, {"high", GeometryQuality::high}};
}

template <>
inline std::vector<std::pair<std::string, DataType>> enumMap()
{
    return {{"float", DataType::FLOAT},   {"double", DataType::DOUBLE}, {"uint8", DataType::UINT8},
            {"uint16", DataType::UINT16}, {"uint32", DataType::UINT32}, {"int8", DataType::INT8},
            {"int16", DataType::INT16},   {"int32", DataType::INT32}};
}
} // namespace core
