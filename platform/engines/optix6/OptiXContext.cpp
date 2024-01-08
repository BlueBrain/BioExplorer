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

#include "OptiXContext.h"
#include "Logs.h"
#include "OptiXCameraProgram.h"
#include "OptiXCommonStructs.h"
#include "OptiXTypes.h"
#include "OptiXUtils.h"

#include <platform/engines/optix6/OptiX6Engine_generated_Cones.cu.ptx.h>
#include <platform/engines/optix6/OptiX6Engine_generated_Cylinders.cu.ptx.h>
#include <platform/engines/optix6/OptiX6Engine_generated_SDFGeometries.cu.ptx.h>
#include <platform/engines/optix6/OptiX6Engine_generated_Spheres.cu.ptx.h>
#include <platform/engines/optix6/OptiX6Engine_generated_Streamlines.cu.ptx.h>
#include <platform/engines/optix6/OptiX6Engine_generated_TriangleMesh.cu.ptx.h>
#include <platform/engines/optix6/OptiX6Engine_generated_Volumes.cu.ptx.h>

#include <platform/core/common/material/Texture2D.h>

namespace
{
static const char* CUDA_SPHERES = OptiX6Engine_generated_Spheres_cu_ptx;
static const char* CUDA_CYLINDERS = OptiX6Engine_generated_Cylinders_cu_ptx;
static const char* CUDA_CONES = OptiX6Engine_generated_Cones_cu_ptx;
static const char* CUDA_SDF_GEOMETRIES = OptiX6Engine_generated_SDFGeometries_cu_ptx;
static const char* CUDA_TRIANGLES_MESH = OptiX6Engine_generated_TriangleMesh_cu_ptx;
static const char* CUDA_VOLUMES = OptiX6Engine_generated_Volumes_cu_ptx;
static const char* CUDA_STREAMLINES = OptiX6Engine_generated_Streamlines_cu_ptx;

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
void textureToOptix(T* ptr_dst, const core::Texture2D& texture, const uint8_t face, const uint8_t mipLevel,
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
            ptr_dst[idx_dst + 3u] = hasAlpha ? rawData[idx_src + 3u] : white<T>();
            idx_dst += 4u;
            idx_src += hasAlpha ? 4u : 3u;
        }
    }
}

RTwrapmode wrapModeToOptix(const core::TextureWrapMode mode)
{
    switch (mode)
    {
    case core::TextureWrapMode::clamp_to_border:
        return RT_WRAP_CLAMP_TO_BORDER;
    case core::TextureWrapMode::clamp_to_edge:
        return RT_WRAP_CLAMP_TO_EDGE;
    case core::TextureWrapMode::mirror:
        return RT_WRAP_MIRROR;
    case core::TextureWrapMode::repeat:
    default:
        return RT_WRAP_REPEAT;
    }
}

} // namespace

#define RT_CHECK_ERROR_NO_CONTEXT(func)                                                       \
    do                                                                                        \
    {                                                                                         \
        RTresult code = func;                                                                 \
        if (code != RT_SUCCESS)                                                               \
            throw std::runtime_error("Optix error in function '" + std::string(#func) + "'"); \
    } while (0)

namespace core
{
namespace engine
{
namespace optix
{
std::unique_ptr<OptiXContext> OptiXContext::_context;

OptiXContext::OptiXContext()
{
    _printSystemInformation();
    _initialize();
}

OptiXContext::~OptiXContext()
{
    _rendererPrograms.clear();
    _cameraPrograms.clear();
    RT_DESTROY_MAP(_optixBoundsPrograms);
    RT_DESTROY_MAP(_optixIntersectionPrograms);
    RT_DESTROY_MAP(_optixTextureSamplers);
    RT_DESTROY(_optixContext);
}

OptiXContext& OptiXContext::get()
{
    if (!_context)
        _context.reset(new OptiXContext);

    return *_context;
}

::optix::Material OptiXContext::createMaterial()
{
    return _optixContext->createMaterial();
}

void OptiXContext::addRenderer(const std::string& name, OptiXShaderProgramPtr program)
{
    _rendererPrograms[name] = program;
}

OptiXShaderProgramPtr OptiXContext::getRenderer(const std::string& name)
{
    auto it = _rendererPrograms.find(name);
    if (it == _rendererPrograms.end())
        throw std::runtime_error("Shader program not found for renderer '" + name + "'");
    return it->second;
}

void OptiXContext::addCamera(const std::string& name, OptiXCameraProgramPtr program)
{
    _cameraPrograms[name] = program;
}

OptiXCameraProgramPtr OptiXContext::getCamera(const std::string& name)
{
    auto it = _cameraPrograms.find(name);
    if (it == _cameraPrograms.end())
        throw std::runtime_error("Camera program not found for camera '" + name + "'");
    return it->second;
}

void OptiXContext::setCamera(const std::string& name)
{
    auto camera = getCamera(name);
    _optixContext->setRayGenerationProgram(0, camera->getRayGenerationProgram());
    _optixContext->setMissProgram(0, camera->getMissProgram());
    _optixContext->setExceptionProgram(0, camera->getExceptionProgram());
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

    const bool createMipmaps = texture->getMipLevels() == 1 && useByte && !texture->isCubeMap();
    uint16_t mipMapLevels = texture->getMipLevels();
    if (createMipmaps)
        mipMapLevels = texture->getPossibleMipMapsLevels();

    if (createMipmaps && !useByte)
        throw std::runtime_error(
            "Non 8-bits textures are not supported for automatic mipmaps "
            "generation");

    RTformat optixFormat = useByte ? RT_FORMAT_UNSIGNED_BYTE4 : RT_FORMAT_FLOAT4;

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
    ::optix::Buffer buffer;
    if (texture->isCubeMap())
        buffer = _optixContext->createCubeBuffer(RT_BUFFER_INPUT, optixFormat, nx, ny, mipMapLevels);
    else
        buffer = _optixContext->createMipmappedBuffer(RT_BUFFER_INPUT, optixFormat, nx, ny, mipMapLevels);

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

        for (uint8_t currentLevel = 1u; currentLevel < mipMapLevels; ++currentLevel)
        {
            ptr_dst = (uint8_t*)mipMapBuffers[currentLevel];
            uint8_t* ptr_src = (uint8_t*)mipMapBuffers[currentLevel - 1u];
            for (uint16_t y = 0u; y < ny; ++y)
            {
                for (uint16_t x = 0u; x < nx; ++x)
                {
                    ptr_dst[(y * nx + x) * 4u] =
                        (ptr_src[(y * 2u * nx + x) * 8u] + ptr_src[((y * 2u * nx + x) * 2u + 1u) * 4u] +
                         ptr_src[((y * 2u + 1u) * nx + x) * 8u] + ptr_src[(((y * 2u + 1u) * nx + x) * 2u + 1u) * 4u]) /
                        4.0f;
                    ptr_dst[(y * nx + x) * 4u + 1u] =
                        (ptr_src[(y * 2u * nx + x) * 8u + 1u] + ptr_src[((y * 2u * nx + x) * 2u + 1u) * 4u + 1u] +
                         ptr_src[((y * 2u + 1u) * nx + x) * 8u + 1u] +
                         ptr_src[(((y * 2u + 1u) * nx + x) * 2u + 1u) * 4u + 1u]) /
                        4.0f;
                    ptr_dst[(y * nx + x) * 4u + 2u] =
                        (ptr_src[(y * 2u * nx + x) * 8u + 2u] + ptr_src[((y * 2u * nx + x) * 2u + 1u) * 4u + 2u] +
                         ptr_src[((y * 2u + 1u) * nx + x) * 8u + 2u] +
                         ptr_src[(((y * 2u + 1u) * nx + x) * 2u + 1u) * 4u + 2u]) /
                        4.0f;
                    ptr_dst[(y * nx + x) * 4u + 3u] =
                        (ptr_src[(y * 2u * nx + x) * 8u + 3u] + ptr_src[((y * 2u * nx + x) * 2u + 1u) * 4u + 3u] +
                         ptr_src[((y * 2u + 1u) * nx + x) * 8u + 3u] +
                         ptr_src[(((y * 2u + 1u) * nx + x) * 2u + 1u) * 4u + 3u]) /
                        4.0f;

                    if (texture->isNormalMap())
                    {
                        glm::vec3 normalized =
                            glm::normalize(glm::vec3(2.0f * (float)ptr_dst[(y * nx + x) * 4u] / 255.0f - 1.0f,
                                                     2.0f * (float)ptr_dst[(y * nx + x) * 4u + 1u] / 255.0f - 1.0f,
                                                     2.0f * (float)ptr_dst[(y * nx + x) * 4u + 2u] / 255.0f - 1.0f));
                        ptr_dst[(y * nx + x) * 4u] = 255.0f * (0.5f * normalized.x + 0.5f);
                        ptr_dst[(y * nx + x) * 4u + 1u] = 255.0f * (0.5f * normalized.y + 0.5f);
                        ptr_dst[(y * nx + x) * 4u + 2u] = 255.0f * (0.5f * normalized.z + 0.5f);
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
                               mipMapLevels > 1 ? RT_FILTER_LINEAR : RT_FILTER_NONE);
    sampler->validate();
    return sampler;
}

void OptiXContext::_initialize()
{
    PLUGIN_DEBUG("Creating context...");
    _optixContext = ::optix::Context::create();

    if (!_optixContext)
        throw(std::runtime_error("Failed to initialize OptiX"));

#ifdef NDEBUG
    _optixContext->setPrintEnabled(false);
#else
    _optixContext->setPrintEnabled(true);
    _optixContext->setPrintBufferSize(1024);
#endif

    _optixContext->setRayTypeCount(OPTIX_RAY_TYPE_COUNT);
    _optixContext->setEntryPointCount(OPTIX_ENTRY_POINT_COUNT);
    _optixContext->setStackSize(OPTIX_STACK_SIZE);
    _optixContext->setMaxTraceDepth(DEFAULT_RENDERER_MAX_RAY_DEPTH);

    _optixBoundsPrograms[OptixGeometryType::cone] =
        _optixContext->createProgramFromPTXString(CUDA_CONES, OPTIX_CUDA_FUNCTION_BOUNDS);
    _optixIntersectionPrograms[OptixGeometryType::cone] =
        _optixContext->createProgramFromPTXString(CUDA_CONES, OPTIX_CUDA_FUNCTION_INTERSECTION);

    _optixBoundsPrograms[OptixGeometryType::cylinder] =
        _optixContext->createProgramFromPTXString(CUDA_CYLINDERS, OPTIX_CUDA_FUNCTION_BOUNDS);
    _optixIntersectionPrograms[OptixGeometryType::cylinder] =
        _optixContext->createProgramFromPTXString(CUDA_CYLINDERS, OPTIX_CUDA_FUNCTION_INTERSECTION);

    _optixBoundsPrograms[OptixGeometryType::sphere] =
        _optixContext->createProgramFromPTXString(CUDA_SPHERES, OPTIX_CUDA_FUNCTION_BOUNDS);
    _optixIntersectionPrograms[OptixGeometryType::sphere] =
        _optixContext->createProgramFromPTXString(CUDA_SPHERES, OPTIX_CUDA_FUNCTION_INTERSECTION);

    _optixBoundsPrograms[OptixGeometryType::triangleMesh] =
        _optixContext->createProgramFromPTXString(CUDA_TRIANGLES_MESH, OPTIX_CUDA_FUNCTION_BOUNDS);
    _optixIntersectionPrograms[OptixGeometryType::triangleMesh] =
        _optixContext->createProgramFromPTXString(CUDA_TRIANGLES_MESH, OPTIX_CUDA_FUNCTION_INTERSECTION);

    _optixBoundsPrograms[OptixGeometryType::volume] =
        _optixContext->createProgramFromPTXString(CUDA_VOLUMES, OPTIX_CUDA_FUNCTION_BOUNDS);
    _optixIntersectionPrograms[OptixGeometryType::volume] =
        _optixContext->createProgramFromPTXString(CUDA_VOLUMES, OPTIX_CUDA_FUNCTION_INTERSECTION);

    _optixBoundsPrograms[OptixGeometryType::streamline] =
        _optixContext->createProgramFromPTXString(CUDA_STREAMLINES, OPTIX_CUDA_FUNCTION_BOUNDS);
    _optixIntersectionPrograms[OptixGeometryType::streamline] =
        _optixContext->createProgramFromPTXString(CUDA_STREAMLINES, OPTIX_CUDA_FUNCTION_INTERSECTION);

    _optixBoundsPrograms[OptixGeometryType::sdfGeometry] =
        _optixContext->createProgramFromPTXString(CUDA_SDF_GEOMETRIES, OPTIX_CUDA_FUNCTION_BOUNDS);
    _optixIntersectionPrograms[OptixGeometryType::sdfGeometry] =
        _optixContext->createProgramFromPTXString(CUDA_SDF_GEOMETRIES, OPTIX_CUDA_FUNCTION_INTERSECTION);

    // Exceptions
    _optixContext[CONTEXT_EXCEPTION_BAD_COLOR]->setFloat(1.0f, 0.0f, 0.0f, 1.f);

    // Volumes
    _optixContext[CONTEXT_VOLUME_SIZE]->setUint(sizeof(VolumeGeometry) / sizeof(float));

    PLUGIN_DEBUG("Context created");
}

void OptiXContext::_printSystemInformation() const
{
    unsigned int optixVersion;
    RT_CHECK_ERROR_NO_CONTEXT(rtGetVersion(&optixVersion));

    unsigned int major = optixVersion / 1000; // Check major with old formula.
    unsigned int minor;
    unsigned int micro;
    if (3 < major) // New encoding since OptiX 4.0.0 to get two digits micro
                   // numbers?
    {
        major = optixVersion / 10000;
        minor = (optixVersion % 10000) / 100;
        micro = optixVersion % 100;
    }
    else // Old encoding with only one digit for the micro number.
    {
        minor = (optixVersion % 1000) / 10;
        micro = optixVersion % 10;
    }
    PLUGIN_INFO("OptiX " << major << "." << minor << "." << micro);

    unsigned int numberOfDevices = 0;
    RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetDeviceCount(&numberOfDevices));
    PLUGIN_INFO("Number of Devices = " << numberOfDevices);

    for (unsigned int i = 0; i < numberOfDevices; ++i)
    {
        char name[256];
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_NAME, sizeof(name), name));
        PLUGIN_INFO("Device " << i << ": " << name);

        int computeCapability[2] = {0, 0};
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY,
                                                       sizeof(computeCapability), &computeCapability));
        PLUGIN_INFO("  Compute Support: " << computeCapability[0] << "." << computeCapability[1]);

        RTsize totalMemory = 0;
        RT_CHECK_ERROR_NO_CONTEXT(
            rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY, sizeof(totalMemory), &totalMemory));
        PLUGIN_INFO("  Total Memory: " << (unsigned long long)(totalMemory / 1024 / 1024) << " MB");

        int clockRate = 0;
        RT_CHECK_ERROR_NO_CONTEXT(
            rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CLOCK_RATE, sizeof(clockRate), &clockRate));
        PLUGIN_INFO("  Clock Rate: " << (clockRate / 1000) << " MHz");

        int maxThreadsPerBlock = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                                       sizeof(maxThreadsPerBlock), &maxThreadsPerBlock));
        PLUGIN_INFO("  Max. Threads per Block: " << maxThreadsPerBlock);

        int smCount = 0;
        RT_CHECK_ERROR_NO_CONTEXT(
            rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, sizeof(smCount), &smCount));
        PLUGIN_INFO("  Streaming Multiprocessor Count: " << smCount);

        int executionTimeoutEnabled = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED,
                                                       sizeof(executionTimeoutEnabled), &executionTimeoutEnabled));
        PLUGIN_INFO("  Execution Timeout Enabled: " << executionTimeoutEnabled);

        int maxHardwareTextureCount = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT,
                                                       sizeof(maxHardwareTextureCount), &maxHardwareTextureCount));
        PLUGIN_INFO("  Max. Hardware Texture Count: " << maxHardwareTextureCount);

        int tccDriver = 0;
        RT_CHECK_ERROR_NO_CONTEXT(
            rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_TCC_DRIVER, sizeof(tccDriver), &tccDriver));
        PLUGIN_INFO("  TCC Driver enabled: " << tccDriver);

        int cudaDeviceOrdinal = 0;
        RT_CHECK_ERROR_NO_CONTEXT(rtDeviceGetAttribute(i, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL,
                                                       sizeof(cudaDeviceOrdinal), &cudaDeviceOrdinal));
        PLUGIN_INFO("  CUDA Device Ordinal: " << cudaDeviceOrdinal);
    }
}

::optix::Geometry OptiXContext::createGeometry(const OptixGeometryType type)
{
    ::optix::Geometry geometry = _optixContext->createGeometry();
    geometry->setBoundingBoxProgram(_optixBoundsPrograms[type]);
    geometry->setIntersectionProgram(_optixIntersectionPrograms[type]);
    return geometry;
}

::optix::GeometryGroup OptiXContext::createGeometryGroup(const bool compact)
{
    auto group = _optixContext->createGeometryGroup();
    auto accel =
        _optixContext->createAcceleration(compact ? OPTIX_ACCELERATION_TYPE_SBVH : DEFAULT_ACCELERATION_STRUCTURE);
    accel->setProperty(OPTIX_ACCELERATION_VERTEX_BUFFER_NAME, "vertices_buffer");
    accel->setProperty(OPTIX_ACCELERATION_VERTEX_BUFFER_STRIDE, "12");
    accel->setProperty(OPTIX_ACCELERATION_INDEX_BUFFER_NAME, "indices_buffer");
    accel->setProperty(OPTIX_ACCELERATION_INDEX_BUFFER_STRIDE, "12");
    group->setAcceleration(accel);
    return group;
}

::optix::Group OptiXContext::createGroup()
{
    auto group = _optixContext->createGroup();
    group->setAcceleration(_optixContext->createAcceleration(DEFAULT_ACCELERATION_STRUCTURE));
    return group;
}
} // namespace optix
} // namespace engine
} // namespace core
