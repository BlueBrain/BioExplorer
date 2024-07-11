/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "MediaMakerPlugin.h"

#include <Defines.h>
#include <Version.h>

#include <plugin/common/Logs.h>
#include <plugin/common/Properties.h>
#include <plugin/common/Types.h>
#include <plugin/common/Utils.h>

#include <plugin/handlers/CameraHandler.h>

#include <platform/core/common/ActionInterface.h>
#include <platform/core/common/Properties.h>
#include <platform/core/common/utils/ImageUtils.h>
#include <platform/core/engineapi/Camera.h>
#include <platform/core/engineapi/Engine.h>
#include <platform/core/engineapi/FrameBuffer.h>
#include <platform/core/engineapi/Model.h>
#include <platform/core/engineapi/Scene.h>
#include <platform/core/parameters/ParametersManager.h>
#include <platform/core/pluginapi/Plugin.h>

#ifdef USE_OPTIX6
#include <MediaMaker_generated_Albedo.cu.ptx.h>
#include <MediaMaker_generated_AmbientOcclusion.cu.ptx.h>
#include <MediaMaker_generated_Depth.cu.ptx.h>
#include <MediaMaker_generated_GeometryNormal.cu.ptx.h>
#include <MediaMaker_generated_Radiance.cu.ptx.h>
#include <MediaMaker_generated_ShadingNormal.cu.ptx.h>
#include <MediaMaker_generated_Shadow.cu.ptx.h>
#include <platform/engines/optix6/OptiXContext.h>
#include <platform/engines/optix6/OptiXProperties.h>
#endif

#include <OpenImageIO/imagebufalgo.h>

#include <fstream>

#include <tiffio.h>

OIIO_NAMESPACE_USING

namespace bioexplorer
{
namespace mediamaker
{
using namespace core;
// Number of floats used to define the camera
const size_t CAMERA_DEFINITION_SIZE = 12;

#define CATCH_STD_EXCEPTION()           \
    catch (const std::runtime_error &e) \
    {                                   \
        response.status = false;        \
        response.contents = e.what();   \
        PLUGIN_ERROR << e.what() );     \
    }

void _addDepthRenderer(core::Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_DEPTH);
    core::PropertyMap properties;
    properties.setProperty(MEDIA_MAKER_RENDERER_PROPERTY_DEPTH_INFINITY);
    engine.addRendererType(RENDERER_DEPTH, properties);
}

void _addAlbedoRenderer(core::Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_ALBEDO);
    core::PropertyMap properties;
    properties.setProperty(RENDERER_PROPERTY_MAX_RAY_DEPTH);
    auto &params = engine.getParametersManager().getApplicationParameters();
    const auto &engineName = params.getEngine();
    if (engineName == ENGINE_OSPRAY)
        properties.setProperty(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER);
    engine.addRendererType(RENDERER_ALBEDO, properties);
}

void _addAmbientOcclusionRenderer(core::Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_AMBIENT_OCCLUSION);
    core::PropertyMap properties;
    auto samples = RENDERER_PROPERTY_GLOBAL_ILLUMINATION_SAMPLES;
    samples.set(16);
    properties.setProperty(samples);
    properties.setProperty(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_RAY_LENGTH);
    properties.setProperty(RENDERER_PROPERTY_MAX_RAY_DEPTH);
    auto &params = engine.getParametersManager().getApplicationParameters();
    const auto &engineName = params.getEngine();
    if (engineName == ENGINE_OSPRAY)
        properties.setProperty(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER);
    properties.setProperty(COMMON_PROPERTY_USE_HARDWARE_RANDOMIZER);
    engine.addRendererType(RENDERER_AMBIENT_OCCLUSION, properties);
}

void _addShadowRenderer(core::Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_SHADOW);
    core::PropertyMap properties;
    properties.setProperty(RENDERER_PROPERTY_SHADOW_SAMPLES);
    properties.setProperty(RENDERER_PROPERTY_GLOBAL_ILLUMINATION_RAY_LENGTH);
    properties.setProperty(RENDERER_PROPERTY_SOFT_SHADOW_STRENGTH);
    engine.addRendererType(RENDERER_SHADOW, properties);
}

void _addRadianceRenderer(core::Engine &engine)
{
    PLUGIN_REGISTER_RENDERER(RENDERER_RADIANCE);
    core::PropertyMap properties;
    engine.addRendererType(RENDERER_RADIANCE, properties);
}

MediaMakerPlugin::MediaMakerPlugin()
    : ExtensionPlugin()
{
}

void MediaMakerPlugin::init()
{
    auto actionInterface = _api->getActionInterface();
    if (actionInterface)
    {
        std::string entryPoint = PLUGIN_API_PREFIX + "version";
        PLUGIN_REGISTER_ENDPOINT(entryPoint);
        actionInterface->registerRequest<Response>(entryPoint, [&]() { return _version(); });

        entryPoint = PLUGIN_API_PREFIX + "set-odu-camera";
        PLUGIN_REGISTER_ENDPOINT(entryPoint);
        actionInterface->registerNotification<CameraDefinition>(entryPoint,
                                                                [&](const CameraDefinition &s) { _setCamera(s); });

        entryPoint = PLUGIN_API_PREFIX + "get-odu-camera";
        PLUGIN_REGISTER_ENDPOINT(entryPoint);
        actionInterface->registerRequest<CameraDefinition>(entryPoint,
                                                           [&]() -> CameraDefinition { return _getCamera(); });

        entryPoint = PLUGIN_API_PREFIX + "export-frames-to-disk";
        PLUGIN_REGISTER_ENDPOINT(entryPoint);
        actionInterface->registerNotification<ExportFramesToDisk>(entryPoint, [&](const ExportFramesToDisk &s)
                                                                  { _exportFramesToDisk(s); });

        entryPoint = PLUGIN_API_PREFIX + "get-export-frames-progress";
        PLUGIN_REGISTER_ENDPOINT(entryPoint);
        actionInterface->registerRequest<FrameExportProgress>(entryPoint,
                                                              [&](void) -> FrameExportProgress
                                                              { return _getFrameExportProgress(); });

        entryPoint = PLUGIN_API_PREFIX + "attach-odu-camera-handler";
        PLUGIN_REGISTER_ENDPOINT(entryPoint);
        actionInterface->registerNotification<CameraHandlerDetails>(entryPoint, [&](const CameraHandlerDetails &s)
                                                                    { _attachCameraHandler(s); });
    }

    auto &engine = _api->getEngine();
    auto &params = engine.getParametersManager().getApplicationParameters();
    const auto &engineName = params.getEngine();
#ifdef USE_OPTIX6
    if (engineName == ENGINE_OPTIX_6)
    {
        _createOptiXRenderers();
        _createRenderers();
    }
#endif
    if (engineName == ENGINE_OSPRAY)
        _createRenderers();
}

#ifdef USE_OPTIX6
void MediaMakerPlugin::_createOptiXRenderers()
{
    std::map<std::string, std::string> renderers = {
        {RENDERER_ALBEDO, MediaMaker_generated_Albedo_cu_ptx},
        {RENDERER_SHADING_NORMAL, MediaMaker_generated_ShadingNormal_cu_ptx},
        {RENDERER_GEOMETRY_NORMAL, MediaMaker_generated_GeometryNormal_cu_ptx},
        {RENDERER_AMBIENT_OCCLUSION, MediaMaker_generated_AmbientOcclusion_cu_ptx},
        {RENDERER_SHADOW, MediaMaker_generated_Shadow_cu_ptx},
        {RENDERER_DEPTH, MediaMaker_generated_Depth_cu_ptx},
        {RENDERER_RADIANCE, MediaMaker_generated_Radiance_cu_ptx},
    };
    ::core::engine::optix::OptiXContext &context = ::core::engine::optix::OptiXContext::get();
    for (const auto &renderer : renderers)
    {
        PLUGIN_REGISTER_RENDERER(renderer.first);
        const std::string ptx = renderer.second;

        auto osp = std::make_shared<::core::engine::optix::OptixShaderProgram>();
        osp->closest_hit = context.getOptixContext()->createProgramFromPTXString(
            ptx, ::core::engine::optix::OPTIX_CUDA_FUNCTION_CLOSEST_HIT_RADIANCE);
        osp->closest_hit_textured = context.getOptixContext()->createProgramFromPTXString(
            ptx, ::core::engine::optix::OPTIX_CUDA_FUNCTION_CLOSEST_HIT_RADIANCE_TEXTURED);
        osp->any_hit = context.getOptixContext()->createProgramFromPTXString(
            ptx, ::core::engine::optix::OPTIX_CUDA_FUNCTION_ANY_HIT_SHADOW);

        context.addRenderer(renderer.first, osp);
    }
}
#endif

void MediaMakerPlugin::_setCamera(const CameraDefinition &payload)
{
    auto &camera = _api->getCamera();
    setCamera(cameraDefinitionToKeyFrame(payload), camera);
}

CameraDefinition MediaMakerPlugin::_getCamera()
{
    auto &camera = _api->getCamera();
    const auto keyFrame = getCameraKeyFrame(camera);
    return keyFrameToCameraDefinition(keyFrame);
}

void MediaMakerPlugin::_createRenderers()
{
    auto &engine = _api->getEngine();
    _addAlbedoRenderer(engine);
    _addDepthRenderer(engine);
    _addAmbientOcclusionRenderer(engine);
    _addShadowRenderer(engine);
    _addRadianceRenderer(engine);
    engine.addRendererType(RENDERER_GEOMETRY_NORMAL);
    engine.addRendererType(RENDERER_SHADING_NORMAL);
}

Response MediaMakerPlugin::_version() const
{
    Response response;
    response.contents = PACKAGE_VERSION_STRING;
    return response;
}

void MediaMakerPlugin::preRender()
{
    if (_exportFramesToDiskDirty && _accumulationFrameNumber == 0)
    {
        auto &frameBuffer = _api->getEngine().getFrameBuffer();
        frameBuffer.resize(_frameBufferSize);
        frameBuffer.clear();

        auto &camera = _api->getCamera();
        if (camera.hasProperty(CAMERA_PROPERTY_ASPECT_RATIO.name))
            camera.updateProperty(CAMERA_PROPERTY_ASPECT_RATIO.name,
                                  static_cast<double>(_frameBufferSize.x) / static_cast<double>(_frameBufferSize.y));
        camera.commit();

        const uint64_t i = CAMERA_DEFINITION_SIZE * _frameNumber;
        // Camera position
        CameraDefinition cd;
        const auto &ci = _exportFramesToDiskPayload.cameraInformation;
        cd.origin = {ci[i], ci[i + 1], ci[i + 2]};
        cd.direction = {ci[i + 3], ci[i + 4], ci[i + 5]};
        cd.up = {ci[i + 6], ci[i + 7], ci[i + 8]};
        cd.apertureRadius = ci[i + 9];
        cd.focalDistance = ci[i + 10];
        cd.interpupillaryDistance = ci[i + 11];
        _setCamera(cd);

        // Animation parameters
        const auto &ai = _exportFramesToDiskPayload.animationInformation;
        if (!ai.empty())
            _api->getParametersManager().getAnimationParameters().setFrame(ai[_frameNumber]);
    }
}

void MediaMakerPlugin::postRender()
{
    ++_accumulationFrameNumber;

    if (_exportFramesToDiskDirty)
    {
        try
        {
            if (_exportFramesToDiskPayload.exportIntermediateFrames)
                _exportFrameToDisk();

            if (_accumulationFrameNumber == _exportFramesToDiskPayload.spp)
            {
                ++_frameNumber;
                _accumulationFrameNumber = 0;
                _exportFramesToDiskDirty = (_frameNumber < _exportFramesToDiskPayload.endFrame);
                _exportFrameToDisk();
            }
        }
        catch (const std::runtime_error &e)
        {
            PLUGIN_ERROR(e.what());
        }
    }
}

void writeBufferToFile(const std::vector<unsigned char> &buffer, const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
        CORE_THROW("Failed to create " + filename);
    file.write(reinterpret_cast<const char *>(buffer.data()), buffer.size());
    file.close();
}

void MediaMakerPlugin::_exportColorBuffer() const
{
    auto &frameBuffer = _api->getEngine().getFrameBuffer();
    auto image = frameBuffer.getImage();
    ImageBuf rotatedBuf;
    ImageBufAlgo::flip(rotatedBuf, image);
    swapRedBlue32(rotatedBuf);

    // Determine the output format
    std::string format = _exportFramesToDiskPayload.format;
    if (format != "jpg" && format != "png" && format != "tiff")
        CORE_THROW("Unknown format: " + format);

    int quality = _exportFramesToDiskPayload.quality;

    // Prepare the filename
    const auto filename = _exportFramesToDiskPayload.path + "/" + _exportFramesToDiskPayload.baseName + "." + format;

    // Set up ImageSpec for output image
    ImageSpec spec = rotatedBuf.spec();
    if (format == "jpg")
        spec.attribute("CompressionQuality", quality);

    // Create an output buffer and write image to memory
    std::vector<unsigned char> buffer(spec.image_bytes());

    auto out = ImageOutput::create(filename);
    if (!out)
        CORE_THROW("Could not create image output.");

    out->open(filename, spec);
    out->write_image(TypeDesc::UINT8, rotatedBuf.localpixels());
    out->close();

    PLUGIN_INFO("Color frame saved to " + filename);
}

void MediaMakerPlugin::_exportDepthBuffer() const
{
    auto &frameBuffer = _api->getEngine().getFrameBuffer();
    frameBuffer.map();
    const auto depthBuffer = frameBuffer.getFloatBuffer();
    const auto &size = frameBuffer.getSize();

    const auto filename = _exportFramesToDiskPayload.path + "/" + _exportFramesToDiskPayload.baseName + "." +
                          _exportFramesToDiskPayload.format;

    TIFF *image = TIFFOpen(filename.c_str(), "w");
    TIFFSetField(image, TIFFTAG_IMAGEWIDTH, size.x);
    TIFFSetField(image, TIFFTAG_IMAGELENGTH, size.y);
    TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, 1);
    TIFFSetField(image, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);

    float *scan_line = (float *)malloc(1 + size.x * (sizeof(float)));

    for (uint32_t i = 0; i < size.y; ++i)
    {
        memcpy(scan_line, &depthBuffer[i * size.x], size.x * sizeof(float));
        TIFFWriteScanline(image, scan_line, size.y - 1 - i, 0);
    }

    TIFFClose(image);
    free(scan_line);

    PLUGIN_INFO("Depth frame saved to " + filename);

    frameBuffer.unmap();
}

void MediaMakerPlugin::_exportFrameToDisk() const
{
    switch (_exportFramesToDiskPayload.frameBufferMode)
    {
    case FrameBufferMode::color:
        _exportColorBuffer();
        break;
    case FrameBufferMode::depth:
        _exportDepthBuffer();
        break;
    default:
        PLUGIN_THROW("Undefined frame buffer mode")
    }
}

void MediaMakerPlugin::_exportFramesToDisk(const ExportFramesToDisk &payload)
{
    _exportFramesToDiskPayload = payload;
    _exportFramesToDiskDirty = true;
    _frameNumber = payload.startFrame;
    _frameBufferSize = Vector2ui(payload.size[0], payload.size[1]);
    _accumulationFrameNumber = 0;
    _baseName = payload.baseName;

    const size_t nbFrames = _exportFramesToDiskPayload.endFrame - _exportFramesToDiskPayload.startFrame;
    PLUGIN_INFO(
        "----------------------------------------------------------------------"
        "----------");
    PLUGIN_INFO("Movie settings               :");
    PLUGIN_INFO("- Samples per pixel          : " + std::to_string(payload.spp));
    PLUGIN_INFO("- Frame size                 : " + std::to_string(_frameBufferSize.x) + "x" +
                std::to_string(_frameBufferSize.y));
    PLUGIN_INFO("- Export folder              : " + payload.path);
    PLUGIN_INFO("- Export intermediate frames : " + std::string(payload.exportIntermediateFrames ? "Yes" : "No"));
    PLUGIN_INFO("- Start frame                : " + std::to_string(payload.startFrame));
    PLUGIN_INFO("- End frame                  : " << std::to_string(payload.endFrame));
    PLUGIN_INFO("- Frame base name            : " << payload.baseName);
    PLUGIN_INFO("- Number of frames           : " << std::to_string(nbFrames));
    PLUGIN_INFO(
        "----------------------------------------------------------------------"
        "----------");
}

FrameExportProgress MediaMakerPlugin::_getFrameExportProgress()
{
    FrameExportProgress result;
    double percentage = 1.f;
    const size_t nbFrames = _exportFramesToDiskPayload.cameraInformation.size() / CAMERA_DEFINITION_SIZE;
    const size_t totalNumberOfFrames = nbFrames * _exportFramesToDiskPayload.spp;

    if (totalNumberOfFrames != 0)
    {
        const double currentProgress = _frameNumber * _exportFramesToDiskPayload.spp + _accumulationFrameNumber;
        percentage = currentProgress / double(totalNumberOfFrames);
    }
    result.progress = percentage;
    result.done = !_exportFramesToDiskDirty;
    PLUGIN_DEBUG("Percentage = " << result.progress << ", Done = " << (result.done ? "True" : "False"));
    return result;
}

void MediaMakerPlugin::_attachCameraHandler(const CameraHandlerDetails &payload)
{
    auto &scene = _api->getScene();
    const auto modelDescriptors = scene.getModelDescriptors();
    if (modelDescriptors.empty())
        PLUGIN_THROW("At least one model is required in the scene");

    if (payload.directions.size() != payload.origins.size())
        PLUGIN_THROW("Invalid number of values for direction vectors");

    if (payload.ups.size() != payload.origins.size())
        PLUGIN_THROW("Invalid number of values for up vectors");

    const uint64_t nbKeyFrames = payload.origins.size() / 3;
    CameraKeyFrames keyFrames;
    for (uint64_t i = 0; i < nbKeyFrames; ++i)
    {
        CameraKeyFrame keyFrame;
        keyFrame.origin = {payload.origins[i * 3], payload.origins[i * 3 + 1], payload.origins[i * 3 + 2]};
        keyFrame.direction = {payload.directions[i * 3], payload.directions[i * 3 + 1], payload.directions[i * 3 + 2]};
        keyFrame.up = {payload.ups[i * 3], payload.ups[i * 3 + 1], payload.ups[i * 3 + 2]};
        keyFrame.apertureRadius = payload.apertureRadii[i];
        keyFrame.focalDistance = payload.focalDistances[i];
        keyFrames.push_back(keyFrame);
    }

    auto modelDescriptor = modelDescriptors[0];
    if (!modelDescriptor)
        PLUGIN_THROW("Invalid model");

    auto &model = modelDescriptor->getModel();
    auto &camera = _api->getCamera();
    auto handler = std::make_shared<CameraHandler>(camera, keyFrames, payload.stepsBetweenKeyFrames,
                                                   payload.numberOfSmoothingSteps);
    model.setSimulationHandler(handler);
}

extern "C" ExtensionPlugin *core_plugin_create(int /*argc*/, char ** /*argv*/)
{
    PLUGIN_INFO("Initializing Media Maker plug-in (version " << PACKAGE_VERSION_STRING << ")");
    PLUGIN_INFO("");
    PLUGIN_INFO("_|      _|                  _|  _|                _|      _|            _|                          ");
    PLUGIN_INFO("_|_|  _|_|    _|_|      _|_|_|        _|_|_|      _|_|  _|_|    _|_|_|  _|  _|      _|_|    _|  _|_|");
    PLUGIN_INFO("_|  _|  _|  _|_|_|_|  _|    _|  _|  _|    _|      _|  _|  _|  _|    _|  _|_|      _|_|_|_|  _|_|    ");
    PLUGIN_INFO("_|      _|  _|        _|    _|  _|  _|    _|      _|      _|  _|    _|  _|  _|    _|        _|      ");
    PLUGIN_INFO("_|      _|    _|_|_|    _|_|_|  _|    _|_|_|      _|      _|    _|_|_|  _|    _|    _|_|_|  _|      ");
    PLUGIN_INFO("");
    return new MediaMakerPlugin();
}

} // namespace mediamaker
} // namespace bioexplorer
