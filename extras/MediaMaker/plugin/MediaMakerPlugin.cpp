/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "MediaMakerPlugin.h"
#include <plugin/common/Logs.h>

#include <brayns/common/ActionInterface.h>
#include <brayns/engineapi/Camera.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/FrameBuffer.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/parameters/ParametersManager.h>
#include <brayns/pluginapi/Plugin.h>

#include <fstream>

#include <tiffio.h>

#include <exiv2/exiv2.hpp>

namespace bioexplorer
{
namespace mediamaker
{
using namespace brayns;

const std::string PLUGIN_API_PREFIX = "mm-";

// Number of floats used to define the camera
const size_t CAMERA_DEFINITION_SIZE = 12;

#define CATCH_STD_EXCEPTION()           \
    catch (const std::runtime_error &e) \
    {                                   \
        response.status = false;        \
        response.contents = e.what();   \
        PLUGIN_ERROR << e.what() );     \
    }

void _addDepthRenderer(brayns::Engine &engine)
{
    PLUGIN_INFO("Registering 'depth' renderer");
    brayns::PropertyMap properties;
    properties.setProperty({"infinity", 1e6, 0., 1e6, {"Infinity"}});
    engine.addRendererType("depth", properties);
}

void _addAlbedoRenderer(brayns::Engine &engine)
{
    PLUGIN_INFO("Registering 'albedo' renderer");
    brayns::PropertyMap properties;
    engine.addRendererType("albedo", properties);
}

void _addAmbientOcclusionRenderer(brayns::Engine &engine)
{
    PLUGIN_INFO("Registering 'ambient occlusion' renderer");
    brayns::PropertyMap properties;
    properties.setProperty(
        {"samplesPerFrame", 16, 1, 256, {"Samples per frame"}});
    properties.setProperty({"rayLength", 1e6, 1e-3, 1e6, {"Ray length"}});
    engine.addRendererType("ambient_occlusion", properties);
}

void _addShadowRenderer(brayns::Engine &engine)
{
    PLUGIN_INFO("Registering 'shadow' renderer");
    brayns::PropertyMap properties;
    properties.setProperty(
        {"samplesPerFrame", 16, 1, 256, {"Samples per frame"}});
    properties.setProperty({"rayLength", 1e6, 1e-3, 1e6, {"Ray length"}});
    properties.setProperty({"softness", 0.0, 0.0, 1.0, {"Shadow softness"}});
    engine.addRendererType("shadow", properties);
}

MediaMakerPlugin::MediaMakerPlugin()
    : ExtensionPlugin()
{
}

void MediaMakerPlugin::init()
{
    auto actionInterface = _api->getActionInterface();
    auto &engine = _api->getEngine();
    auto &params = engine.getParametersManager().getApplicationParameters();
    const auto &engineName = params.getEngine();
    if (engineName != ENGINE_OSPRAY)
        PLUGIN_THROW("OSPRay engine is the only one supported by this plugin")

    if (actionInterface)
    {
        std::string entryPoint = PLUGIN_API_PREFIX + "version";
        PLUGIN_INFO("Registering '" + entryPoint + "' endpoint");
        actionInterface->registerRequest<Response>(entryPoint, [&]()
                                                   { return _version(); });

        entryPoint = PLUGIN_API_PREFIX + "set-odu-camera";
        PLUGIN_INFO("Registering '" + entryPoint + "' endpoint");
        actionInterface->registerNotification<CameraDefinition>(
            entryPoint, [&](const CameraDefinition &s) { _setCamera(s); });

        entryPoint = PLUGIN_API_PREFIX + "get-odu-camera";
        PLUGIN_INFO("Registering '" + entryPoint + "' endpoint");
        actionInterface->registerRequest<CameraDefinition>(
            entryPoint, [&]() -> CameraDefinition { return _getCamera(); });

        entryPoint = PLUGIN_API_PREFIX + "export-frames-to-disk";
        PLUGIN_INFO("Registering '" + entryPoint + "' endpoint");
        actionInterface->registerNotification<ExportFramesToDisk>(
            entryPoint,
            [&](const ExportFramesToDisk &s) { _exportFramesToDisk(s); });

        entryPoint = PLUGIN_API_PREFIX + "get-export-frames-progress";
        PLUGIN_INFO("Registering '" + entryPoint + "' endpoint");
        actionInterface->registerRequest<FrameExportProgress>(
            entryPoint,
            [&](void) -> FrameExportProgress
            { return _getFrameExportProgress(); });
    }

    _addDepthRenderer(engine);
    _addAlbedoRenderer(engine);
    _addAmbientOcclusionRenderer(engine);
    _addShadowRenderer(engine);

    engine.addRendererType("raycast_Ng");
    engine.addRendererType("raycast_Ns");
}

Response MediaMakerPlugin::_version() const
{
    Response response;
    response.contents = PACKAGE_VERSION;
    return response;
}

void MediaMakerPlugin::preRender()
{
    if (_exportFramesToDiskDirty && _accumulationFrameNumber == 0)
    {
        const uint64_t i = CAMERA_DEFINITION_SIZE * _frameNumber;
        // Camera position
        CameraDefinition cd;
        const auto &ci = _exportFramesToDiskPayload.cameraInformation;
        cd.origin = {ci[i], ci[i + 1], ci[i + 2]};
        cd.direction = {ci[i + 3], ci[i + 4], ci[i + 5]};
        cd.up = {ci[i + 6], ci[i + 7], ci[i + 8]};
        cd.apertureRadius = ci[i + 9];
        cd.focusDistance = ci[i + 10];
        cd.interpupillaryDistance = ci[i + 11];
        _setCamera(cd);

        // Animation parameters
        const auto &ai = _exportFramesToDiskPayload.animationInformation;
        if (!ai.empty())
            _api->getParametersManager().getAnimationParameters().setFrame(
                ai[_frameNumber]);
    }
}

void MediaMakerPlugin::postRender()
{
    ++_accumulationFrameNumber;

    if (_exportFramesToDiskDirty)
    {
        if (_exportFramesToDiskPayload.exportIntermediateFrames)
            _exportFrameToDisk();

        if (_accumulationFrameNumber == _exportFramesToDiskPayload.spp)
        {
            _exportFrameToDisk();
            ++_frameNumber;
            _accumulationFrameNumber = 0;
            _exportFramesToDiskDirty =
                (_frameNumber < _exportFramesToDiskPayload.endFrame);
        }
    }
}

void MediaMakerPlugin::_setCamera(const CameraDefinition &payload)
{
    auto &camera = _api->getCamera();

    // Origin
    const auto &o = payload.origin;
    brayns::Vector3d origin{o[0], o[1], o[2]};
    camera.setPosition(origin);

    // Target
    const auto &d = payload.direction;
    brayns::Vector3d direction{d[0], d[1], d[2]};
    camera.setTarget(origin + direction);

    // Up
    const auto &u = payload.up;
    brayns::Vector3d up{u[0], u[1], u[2]};

    // Orientation
    const auto q = glm::inverse(glm::lookAt(origin, origin + direction,
                                            up)); // Not quite sure why this
                                                  // should be inverted?!?
    camera.setOrientation(q);

    // Aperture
    camera.updateProperty("apertureRadius", payload.apertureRadius);

    // Focus distance
    camera.updateProperty("focusDistance", payload.focusDistance);

    // Stereo
    camera.updateProperty("stereo", payload.interpupillaryDistance != 0.0);
    camera.updateProperty("interpupillaryDistance",
                          payload.interpupillaryDistance);

    _api->getCamera().markModified();
}

CameraDefinition MediaMakerPlugin::_getCamera()
{
    const auto &camera = _api->getCamera();

    CameraDefinition cd;
    const auto &p = camera.getPosition();
    cd.origin = {p.x, p.y, p.z};
    const auto d =
        glm::rotate(camera.getOrientation(), brayns::Vector3d(0., 0., -1.));
    cd.direction = {d.x, d.y, d.z};
    const auto u =
        glm::rotate(camera.getOrientation(), brayns::Vector3d(0., 1., 0.));
    cd.up = {u.x, u.y, u.z};
    return cd;
}

const std::string MediaMakerPlugin::_getFileName(
    const std::string &format) const
{
    std::string baseName = _baseName;
    if (baseName.empty())
    {
        char frame[7];
        sprintf(frame, "%05d", _frameNumber);
        baseName = frame;
    }
    return _exportFramesToDiskPayload.path + '/' + baseName + "." + format;
}

void MediaMakerPlugin::_exportColorBuffer() const
{
    auto &frameBuffer = _api->getEngine().getFrameBuffer();
    auto image = frameBuffer.getImage();
    auto fif = _exportFramesToDiskPayload.format == "jpg"
                   ? FIF_JPEG
                   : FreeImage_GetFIFFromFormat(
                         _exportFramesToDiskPayload.format.c_str());
    if (fif == FIF_JPEG)
        image.reset(FreeImage_ConvertTo24Bits(image.get()));
    else if (fif == FIF_UNKNOWN)
        PLUGIN_THROW("Unknown format: " + _exportFramesToDiskPayload.format);

    int flags = _exportFramesToDiskPayload.quality;
    if (fif == FIF_TIFF)
        flags = TIFF_NONE;

    brayns::freeimage::MemoryPtr memory(FreeImage_OpenMemory());

    FreeImage_SaveToMemory(fif, image.get(), memory.get(), flags);

    BYTE *pixels = nullptr;
    DWORD numPixels = 0;
    FreeImage_AcquireMemory(memory.get(), &pixels, &numPixels);

    const auto filename = _getFileName(_exportFramesToDiskPayload.format);
    std::ofstream file;
    file.open(filename, std::ios_base::binary);
    if (!file.is_open())
        PLUGIN_THROW("Failed to create " + filename);

    file.write((char *)pixels, numPixels);
    file.close();
    frameBuffer.clear();

    auto finalImage = Exiv2::ImageFactory::open(filename);
    if (finalImage.get())
    {
        Exiv2::XmpData xmpData;
        xmpData["Xmp.dc.Source"] = "Blue Brain BioExplorer";
        xmpData["Xmp.dc.Subject"] = _exportFramesToDiskPayload.keywords;
        finalImage->setXmpData(xmpData);
        finalImage->writeMetadata();
    }

    PLUGIN_INFO("Color frame saved to " + filename);
}

void MediaMakerPlugin::_exportDepthBuffer() const
{
    auto &frameBuffer = _api->getEngine().getFrameBuffer();
    frameBuffer.map();
    const auto depthBuffer = frameBuffer.getDepthBuffer();
    const auto &size = frameBuffer.getSize();

    const auto filename = _getFileName("tiff");

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
    _accumulationFrameNumber = 0;
    _baseName = payload.baseName;

    auto &frameBuffer = _api->getEngine().getFrameBuffer();
    const auto size = Vector2ui(payload.size[0], payload.size[1]);
    frameBuffer.resize(size);
    frameBuffer.clear();

    auto &camera = _api->getCamera();
    if (camera.hasProperty("aspect"))
        camera.updateProperty("aspect", static_cast<double>(size.x) /
                                            static_cast<double>(size.y));
    camera.commit();
    const size_t nbFrames = _exportFramesToDiskPayload.endFrame -
                            _exportFramesToDiskPayload.startFrame;
    PLUGIN_INFO(
        "----------------------------------------------------------------------"
        "----------");
    PLUGIN_INFO("Movie settings               :");
    PLUGIN_INFO("- Samples per pixel          : " +
                std::to_string(payload.spp));
    PLUGIN_INFO("- Frame size                 : " + std::to_string(size.x) +
                "x" + std::to_string(size.y));
    PLUGIN_INFO("- Export folder              : " + payload.path);
    PLUGIN_INFO("- Export intermediate frames : " +
                std::string(payload.exportIntermediateFrames ? "Yes" : "No"));
    PLUGIN_INFO("- Start frame                : " +
                std::to_string(payload.startFrame));
    PLUGIN_INFO(
        "- End frame                  : " << std::to_string(payload.endFrame));
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
    const size_t nbFrames =
        _exportFramesToDiskPayload.cameraInformation.size() /
        CAMERA_DEFINITION_SIZE;
    const size_t totalNumberOfFrames =
        nbFrames * _exportFramesToDiskPayload.spp;

    if (totalNumberOfFrames != 0)
    {
        const double currentProgress =
            _frameNumber * _exportFramesToDiskPayload.spp +
            _accumulationFrameNumber;
        percentage = currentProgress / double(totalNumberOfFrames);
    }
    result.progress = percentage;
    result.done = !_exportFramesToDiskDirty;
    PLUGIN_DEBUG("Percentage = " << result.progress << ", Done = "
                                 << (result.done ? "True" : "False"));
    return result;
}

extern "C" ExtensionPlugin *brayns_plugin_create(int /*argc*/, char ** /*argv*/)
{
    PLUGIN_INFO("Initializing Media Maker plug-in (version " << PACKAGE_VERSION
                                                             << ")");
    PLUGIN_INFO("");
    PLUGIN_INFO(
        "_|      _|                  _|  _|                _|      "
        "_|            _|                          ");
    PLUGIN_INFO(
        "_|_|  _|_|    _|_|      _|_|_|        _|_|_|      _|_|  "
        "_|_|    _|_|_|  _|  _|      _|_|    _|  _|_|");
    PLUGIN_INFO(
        "_|  _|  _|  _|_|_|_|  _|    _|  _|  _|    _|      _|  _|  "
        "_|  _|    _|  _|_|      _|_|_|_|  _|_|    ");
    PLUGIN_INFO(
        "_|      _|  _|        _|    _|  _|  _|    _|      _|      "
        "_|  _|    _|  _|  _|    _|        _|      ");
    PLUGIN_INFO(
        "_|      _|    _|_|_|    _|_|_|  _|    _|_|_|      _|      "
        "_|    _|_|_|  _|    _|    _|_|_|  _|      ");
    PLUGIN_INFO("");
    return new MediaMakerPlugin();
}

} // namespace mediamaker
} // namespace bioexplorer