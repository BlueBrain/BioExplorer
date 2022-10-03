/* Copyright (c) 2015-2018, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include "OptiXContext.h"
#include "Logs.h"
#include "OptiXCamera.h"

#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <Exception.h>
#include <sutil.h>

// #include <engines/optix/braynsOptixEngine_generated_Cones.cu.ptx.h>
// #include <engines/optix/braynsOptixEngine_generated_Cylinders.cu.ptx.h>
// #include <engines/optix/braynsOptixEngine_generated_Spheres.cu.ptx.h>
// #include <engines/optix/braynsOptixEngine_generated_TriangleMesh.cu.ptx.h>

// #include <brayns/common/material/Texture2D.h>

#if 0
namespace
{
template <typename T>
T white();

template <>
uint8 white()
{
    return 255;
}

template <>
float white()
{
    return 1.f;
}

template <typename T>
void textureToOptix(T* ptr_dst, const brayns::Texture2D& texture,
                    const uint8_t face, const uint8_t mipLevel,
                    const bool hasAlpha)
{
    uint16_t width = texture.width;
    uint16_t height = texture.height;
    for (uint8_t i = 0; i < mipLevel; ++i)
    {
        width /= 2;
        height /= 2;
    }
    size_t idx_src = 0;
    size_t idx_dst = 0;
    const auto rawData = texture.getRawData<T>(face, mipLevel);
    for (uint16_t y = 0; y < height; ++y)
    {
        for (uint16_t x = 0; x < width; ++x)
        {
            ptr_dst[idx_dst] = rawData[idx_src];
            ptr_dst[idx_dst + 1u] = rawData[idx_src + 1u];
            ptr_dst[idx_dst + 2u] = rawData[idx_src + 2u];
            ptr_dst[idx_dst + 3u] =
                hasAlpha ? rawData[idx_src + 3u] : white<T>();
            idx_dst += 4u;
            idx_src += hasAlpha ? 4u : 3u;
        }
    }
}

RTwrapmode wrapModeToOptix(const brayns::TextureWrapMode mode)
{
    switch (mode)
    {
    case brayns::TextureWrapMode::clamp_to_border:
        return RT_WRAP_CLAMP_TO_BORDER;
    case brayns::TextureWrapMode::clamp_to_edge:
        return RT_WRAP_CLAMP_TO_EDGE;
    case brayns::TextureWrapMode::mirror:
        return RT_WRAP_MIRROR;
    case brayns::TextureWrapMode::repeat:
    default:
        return RT_WRAP_REPEAT;
    }
}
} // namespace
#endif

#define RT_CHECK_ERROR_NO_CONTEXT(func)                                     \
    do                                                                      \
    {                                                                       \
        RTresult code = func;                                               \
        if (code != RT_SUCCESS)                                             \
            PLUGIN_THROW("Optix error in function '" + std::string(#func) + \
                         "'");                                              \
    } while (0)

namespace brayns
{
OptiXContext* OptiXContext::_instance = nullptr;
std::mutex OptiXContext::_mutex;

OptiXContext::OptiXContext()
{
    _initialize();
}

OptiXContext::~OptiXContext()
{
    PLUGIN_DEBUG("Destroying OptiX Context");

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state.sbt.missRecordBase)));
    CUDA_CHECK(
        cudaFree(reinterpret_cast<void*>(_state.sbt.hitgroupRecordBase)));

    OPTIX_CHECK(optixProgramGroupDestroy(_state.raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(_state.miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(_state.hitgroup_prog_group));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(_state.stream)));

    OPTIX_CHECK(optixPipelineDestroy(_state.pipeline));

    OPTIX_CHECK(optixDeviceContextDestroy(_state.context));
}

#if 0
::optix::Material OptiXContext::createMaterial()
{
    return _optixContext->createMaterial();
}

void OptiXContext::addRenderer(const std::string& name,
                               OptiXShaderProgramPtr program)
{
    _rendererProgram[name] = program;
}

OptiXShaderProgramPtr OptiXContext::getRenderer(const std::string& name)
{
    auto it = _rendererProgram.find(name);
    if (it == _rendererProgram.end())
        throw std::runtime_error("Shader program not found for renderer '" +
                                 name + "'");
    return it->second;
}

::optix::TextureSampler OptiXContext::createTextureSampler(Texture2DPtr texture)
{
    uint16_t nx = texture->width;
    uint16_t ny = texture->height;
    const uint16_t channels = texture->channels;
    const uint16_t optixChannels = 4;
    const bool hasAlpha = optixChannels == channels;

    const bool useFloat = texture->depth == 4;
    const bool useByte = texture->depth == 1;

    if (!useFloat && !useByte)
        throw std::runtime_error("Only byte or float textures are supported");

    const bool createMipmaps =
        texture->getMipLevels() == 1 && useByte && !texture->isCubeMap();
    uint16_t mipMapLevels = texture->getMipLevels();
    if (createMipmaps)
        mipMapLevels = texture->getPossibleMipMapsLevels();

    if (createMipmaps && !useByte)
        throw std::runtime_error(
            "Non 8-bits textures are not supported for automatic mipmaps "
            "generation");

    RTformat optixFormat =
        useByte ? RT_FORMAT_UNSIGNED_BYTE4 : RT_FORMAT_FLOAT4;

    // Create texture sampler
    ::optix::TextureSampler sampler = _optixContext->createTextureSampler();
    const auto wrapMode = wrapModeToOptix(texture->getWrapMode());
    sampler->setWrapMode(0, wrapMode);
    sampler->setWrapMode(1, wrapMode);
    sampler->setWrapMode(2, wrapMode);
    sampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    sampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    sampler->setMaxAnisotropy(8.0f);

    // Create buffer and populate with texture data
    optix::Buffer buffer;
    if (texture->isCubeMap())
        buffer = _optixContext->createCubeBuffer(RT_BUFFER_INPUT, optixFormat,
                                                 nx, ny, mipMapLevels);
    else
        buffer =
            _optixContext->createMipmappedBuffer(RT_BUFFER_INPUT, optixFormat,
                                                 nx, ny, mipMapLevels);

    std::vector<void*> mipMapBuffers(mipMapLevels);
    for (uint8_t currentLevel = 0u; currentLevel < mipMapLevels; ++currentLevel)
        mipMapBuffers[currentLevel] = buffer->map(currentLevel);

    if (createMipmaps)
    {
        uint8_t* ptr_dst = (uint8_t*)mipMapBuffers[0];
        size_t idx_src = 0;
        size_t idx_dst = 0;
        const auto rawData = texture->getRawData<unsigned char>();
        for (uint16_t y = 0; y < ny; ++y)
        {
            for (uint16_t x = 0; x < nx; ++x)
            {
                ptr_dst[idx_dst] = rawData[idx_src];
                ptr_dst[idx_dst + 1u] = rawData[idx_src + 1u];
                ptr_dst[idx_dst + 2u] = rawData[idx_src + 2u];
                ptr_dst[idx_dst + 3u] = hasAlpha ? rawData[idx_src + 3u] : 255u;
                idx_dst += 4u;
                idx_src += hasAlpha ? 4u : 3u;
            }
        }
        ny /= 2u;
        nx /= 2u;

        for (uint8_t currentLevel = 1u; currentLevel < mipMapLevels;
             ++currentLevel)
        {
            ptr_dst = (uint8_t*)mipMapBuffers[currentLevel];
            uint8_t* ptr_src = (uint8_t*)mipMapBuffers[currentLevel - 1u];
            for (uint16_t y = 0u; y < ny; ++y)
            {
                for (uint16_t x = 0u; x < nx; ++x)
                {
                    ptr_dst[(y * nx + x) * 4u] =
                        (ptr_src[(y * 2u * nx + x) * 8u] +
                         ptr_src[((y * 2u * nx + x) * 2u + 1u) * 4u] +
                         ptr_src[((y * 2u + 1u) * nx + x) * 8u] +
                         ptr_src[(((y * 2u + 1u) * nx + x) * 2u + 1u) * 4u]) /
                        4.0f;
                    ptr_dst[(y * nx + x) * 4u + 1u] =
                        (ptr_src[(y * 2u * nx + x) * 8u + 1u] +
                         ptr_src[((y * 2u * nx + x) * 2u + 1u) * 4u + 1u] +
                         ptr_src[((y * 2u + 1u) * nx + x) * 8u + 1u] +
                         ptr_src[(((y * 2u + 1u) * nx + x) * 2u + 1u) * 4u +
                                 1u]) /
                        4.0f;
                    ptr_dst[(y * nx + x) * 4u + 2u] =
                        (ptr_src[(y * 2u * nx + x) * 8u + 2u] +
                         ptr_src[((y * 2u * nx + x) * 2u + 1u) * 4u + 2u] +
                         ptr_src[((y * 2u + 1u) * nx + x) * 8u + 2u] +
                         ptr_src[(((y * 2u + 1u) * nx + x) * 2u + 1u) * 4u +
                                 2u]) /
                        4.0f;
                    ptr_dst[(y * nx + x) * 4u + 3u] =
                        (ptr_src[(y * 2u * nx + x) * 8u + 3u] +
                         ptr_src[((y * 2u * nx + x) * 2u + 1u) * 4u + 3u] +
                         ptr_src[((y * 2u + 1u) * nx + x) * 8u + 3u] +
                         ptr_src[(((y * 2u + 1u) * nx + x) * 2u + 1u) * 4u +
                                 3u]) /
                        4.0f;

                    if (texture->isNormalMap())
                    {
                        glm::vec3 normalized = glm::normalize(glm::vec3(
                            2.0f * (float)ptr_dst[(y * nx + x) * 4u] / 255.0f -
                                1.0f,
                            2.0f * (float)ptr_dst[(y * nx + x) * 4u + 1u] /
                                    255.0f -
                                1.0f,
                            2.0f * (float)ptr_dst[(y * nx + x) * 4u + 2u] /
                                    255.0f -
                                1.0f));
                        ptr_dst[(y * nx + x) * 4u] =
                            255.0f * (0.5f * normalized.x + 0.5f);
                        ptr_dst[(y * nx + x) * 4u + 1u] =
                            255.0f * (0.5f * normalized.y + 0.5f);
                        ptr_dst[(y * nx + x) * 4u + 2u] =
                            255.0f * (0.5f * normalized.z + 0.5f);
                    }
                }
            }
            ny /= 2u;
            nx /= 2u;
        }
    }
    else
    {
        for (uint8_t face = 0; face < texture->getNumFaces(); ++face)
        {
            auto mipWidth = nx;
            auto mipHeight = ny;
            for (uint16_t mip = 0; mip < mipMapLevels; ++mip)
            {
                if (useByte)
                {
                    auto dst = (uint8_t*)mipMapBuffers[mip];
                    dst += face * mipWidth * mipHeight * 4;
                    textureToOptix<uint8_t>(dst, *texture, face, mip, hasAlpha);
                }
                else if (useFloat)
                {
                    auto dst = (float*)mipMapBuffers[mip];
                    dst += face * mipWidth * mipHeight * 4;
                    textureToOptix<float>(dst, *texture, face, mip, hasAlpha);
                }
                mipWidth /= 2;
                mipHeight /= 2;
            }
        }
    }

    for (uint8_t currentLevel = 0u; currentLevel < mipMapLevels; ++currentLevel)
        buffer->unmap(currentLevel);

    // Assign buffer to sampler
    sampler->setBuffer(buffer);
    sampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR,
                               mipMapLevels > 1 ? RT_FILTER_LINEAR
                                                : RT_FILTER_NONE);
    sampler->validate();
    return sampler;
}
#endif
void OptiXContext::addCamera(const std::string& name, OptiXCameraPtr camera)
{
    _cameras[name] = camera;
}

OptiXCameraPtr OptiXContext::getCamera(const std::string& name)
{
    auto it = _cameras.find(name);
    if (it == _cameras.end())
        PLUGIN_THROW("Camera not found for '" + name + "'");
    return it->second;
}

void OptiXContext::setCamera(const std::string& name)
{
    auto it = _cameras.find(name);
    if (it == _cameras.end())
        PLUGIN_THROW("Camera not found for '" + name + "'");
    _currentCamera = name;
}

void OptiXContext::_createCameraModules()
{
    PLUGIN_DEBUG("Registering OptiX Camera Modules");
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    size_t inputSize = 0;
    const char* input =
        sutil::getInputData(BRAYNS_OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR,
                            "PerspectiveCamera.cu", inputSize);

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(_state.context,
                                             &_state.module_compile_options,
                                             &_state.pipeline_compile_options,
                                             input, inputSize, log, &sizeof_log,
                                             &_state.camera_module));
}

void OptiXContext::_createCameraPrograms()
{
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    // ---------------------------------------------------------------------------------------------
    // Raygen program record
    // ---------------------------------------------------------------------------------------------
    PLUGIN_DEBUG("Registering OptiX Camera Ray Generation Program");
    OptixProgramGroupDesc raygen_prog_group_desc = {}; //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = _state.camera_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(_state.context, &raygen_prog_group_desc,
                                1, // num program groups
                                &_state.program_group_options, log, &sizeof_log,
                                &_state.raygen_prog_group));
    _programGroups.push_back(_state.raygen_prog_group);

    // ---------------------------------------------------------------------------------------------
    // Miss program
    // ---------------------------------------------------------------------------------------------
    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = _state.camera_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__constant_bg";

    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(_state.context, &miss_prog_group_desc, 1,
                                &_state.program_group_options, log, &sizeof_log,
                                &_state.miss_prog_group));
    _programGroups.push_back(_state.miss_prog_group);

    miss_prog_group_desc.miss = {
        nullptr, // module
        nullptr  // entryFunctionName
    };
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(_state.context, &miss_prog_group_desc, 1,
                                &_state.program_group_options, log, &sizeof_log,
                                &_state.occlusion_prog_group));

    _programGroups.push_back(_state.occlusion_prog_group);
}

void OptiXContext::_createShadingModules()
{
    PLUGIN_DEBUG("Creating OptiX Shading Modules");

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    size_t inputSize = 0;
    const char* input =
        sutil::getInputData(BRAYNS_OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR,
                            "Material.cu", inputSize);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(_state.context,
                                             &_state.module_compile_options,
                                             &_state.pipeline_compile_options,
                                             input, inputSize, log, &sizeof_log,
                                             &_state.shading_module));
}

void OptiXContext::_createMaterialPrograms()
{
    PLUGIN_DEBUG("Creating OptiX Material Programs");

    char log[2048];
    size_t sizeof_log = sizeof(log);

    // Radiance
    OptixProgramGroupDesc radiance_prog_group_desc = {};
    radiance_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    radiance_prog_group_desc.hitgroup.moduleIS =
        _state.geometry_module; // Why sphere module, and not shading?
    radiance_prog_group_desc.hitgroup.entryFunctionNameIS =
        "__intersection__sphere";
    radiance_prog_group_desc.hitgroup.moduleCH = _state.shading_module;
    radiance_prog_group_desc.hitgroup.entryFunctionNameCH =
        "__closesthit__radiance";
    radiance_prog_group_desc.hitgroup.moduleAH = nullptr;
    radiance_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(_state.context, &radiance_prog_group_desc, 1,
                                &_state.program_group_options, log, &sizeof_log,
                                &_state.radiance_prog_group));
    _programGroups.push_back(_state.radiance_prog_group);

    // Occlusion
    OptixProgramGroupDesc occlusion_prog_group_desc = {};
    occlusion_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
    occlusion_prog_group_desc.hitgroup.moduleIS = _state.geometry_module;
    occlusion_prog_group_desc.hitgroup.entryFunctionNameIS =
        "__intersection__sphere";
    occlusion_prog_group_desc.hitgroup.moduleCH = _state.shading_module;
    occlusion_prog_group_desc.hitgroup.entryFunctionNameCH =
        "__closesthit__full_occlusion";
    occlusion_prog_group_desc.hitgroup.moduleAH = nullptr;
    occlusion_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(_state.context, &occlusion_prog_group_desc, 1,
                                &_state.program_group_options, log, &sizeof_log,
                                &_state.occlusion_prog_group));
    _programGroups.push_back(_state.occlusion_prog_group);

    // Phong Sphere

    // TODO: REMOVE
    const GeometryData::Sphere g_sphere = {
        {0.f, 0.f, 0.f}, // center
        0.25f            // radius
    };
    // TODO: REMOVE

    const size_t count_records = RAY_TYPE_COUNT * 1; // OBJ_COUNT;
    HitGroupRecord hitgroup_records[count_records];

    // Note: Fill SBT record array the same order like AS is built.
    int sbt_idx = 0;

    // Radiance
    OPTIX_CHECK(optixSbtRecordPackHeader(_state.radiance_prog_group,
                                         &hitgroup_records[sbt_idx]));
    hitgroup_records[sbt_idx].data.geometry.sphere = g_sphere;
    hitgroup_records[sbt_idx].data.shading.phong = {
        {0.2f, 0.5f, 0.5f}, // Ka
        {0.2f, 0.7f, 0.8f}, // Kd
        {0.9f, 0.9f, 0.9f}, // Ks
        {0.5f, 0.5f, 0.5f}, // Kr
        64,                 // phong_exp
    };
    sbt_idx++;

    // Occlusion
    OPTIX_CHECK(optixSbtRecordPackHeader(_state.occlusion_prog_group,
                                         &hitgroup_records[sbt_idx]));
    hitgroup_records[sbt_idx].data.geometry.sphere = g_sphere;

    CUdeviceptr d_hitgroup_records;
    size_t sizeof_hitgroup_record = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records),
                          sizeof_hitgroup_record * count_records));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records),
                          hitgroup_records,
                          sizeof_hitgroup_record * count_records,
                          cudaMemcpyHostToDevice));

    _state.sbt.hitgroupRecordBase = d_hitgroup_records;
    _state.sbt.hitgroupRecordCount = count_records;
    _state.sbt.hitgroupRecordStrideInBytes =
        static_cast<uint32_t>(sizeof_hitgroup_record);
}

void OptiXContext::_createGeometryModules()
{
    PLUGIN_DEBUG("Creating OptiX Geometry Modules");
    size_t inputSize = 0;
    const char* input =
        sutil::getInputData(BRAYNS_OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR,
                            "Spheres.cu", inputSize);
    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(_state.context,
                                             &_state.module_compile_options,
                                             &_state.pipeline_compile_options,
                                             input, inputSize, log, &sizeof_log,
                                             &_state.geometry_module));
}

void OptiXContext::_createGeometryPrograms()
{
    PLUGIN_DEBUG("Creating OptiX Geometry Programs");
    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleIS = _state.geometry_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS =
        "__intersection__sphere";
    hitgroup_prog_group_desc.hitgroup.moduleCH = _state.geometry_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleAH = nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
    OPTIX_CHECK_LOG(
        optixProgramGroupCreate(_state.context, &hitgroup_prog_group_desc,
                                1, // num program groups
                                &_state.program_group_options, log, &sizeof_log,
                                &_state.hitgroup_prog_group));
    _programGroups.push_back(_state.hitgroup_prog_group);
}

void OptiXContext::linkPipeline()
{
    if (_pipelineInitialized)
        return;
    PLUGIN_DEBUG("Linking OptiX Pipeline");
    char log[2048];
    size_t sizeof_log = sizeof(log);
    const uint32_t max_trace_depth = 1;

    _state.pipeline_link_options.maxTraceDepth = max_trace_depth;
#if !defined(NDEBUG)
    _state.pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    PLUGIN_DEBUG("Registering " << _programGroups.size()
                                << " OptiX Programs in the Pipeline");
    OPTIX_CHECK(
        optixPipelineCreate(_state.context, &_state.pipeline_compile_options,
                            &_state.pipeline_link_options,
                            _programGroups.data(),
                            static_cast<unsigned int>(_programGroups.size()),
                            log, &sizeof_log, &_state.pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto& programGroup : _programGroups)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(programGroup, &stack_sizes));

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(
        optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                   0, // maxCCDepth
                                   0, // maxDCDEpth
                                   &direct_callable_stack_size_from_traversal,
                                   &direct_callable_stack_size_from_state,
                                   &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(
        _state.pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        1 // maxTraversableDepth
        ));
    _pipelineInitialized = true;
}

void OptiXContext::_initialize()
{
    PLUGIN_DEBUG("Creating OptiX Context");

    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    // Module compile options
    _state.module_compile_options = {};
#if !defined(NDEBUG)
    _state.module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    _state.module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    // Pipeline compile options
    _state.pipeline_compile_options = {
        false,
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
        5,
        5,
        OPTIX_EXCEPTION_FLAG_NONE,
        "params"};
    _state.pipeline_compile_options.usesMotionBlur = false;
    _state.pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    _state.pipeline_compile_options.numPayloadValues = 5;
    _state.pipeline_compile_options.numAttributeValues = 5;
    _state.pipeline_compile_options.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_NONE; // TODO: should be
                                   // OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    _state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    CUcontext cuCtx = 0; // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &_state.context));

    CUDA_CHECK(cudaStreamCreate(&_state.stream));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&_state.d_params), sizeof(Params)));

    // Modules
    _createCameraModules();
    _createShadingModules();
    _createGeometryModules();

    // Programs attached to modules
    _createCameraPrograms();
    _createMaterialPrograms();
    _createGeometryPrograms();
}

#if 0
Geometry OptiXContext::createGeometry(const OptixGeometryType type)
{
    ::optix::Geometry geometry = _optixContext->createGeometry();
    geometry->setBoundingBoxProgram(_bounds[type]);
    geometry->setIntersectionProgram(_intersects[type]);
    return geometry;
}
#endif

#if 0
Group OptiXContext::createGroup()
{
    auto group = _optixContext->createGroup();
    group->setAcceleration(
        _optixContext->createAcceleration(DEFAULT_ACCELERATION_STRUCTURE));
    return group;
}
#endif
} // namespace brayns
