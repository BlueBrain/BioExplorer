/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#include "MediaMakerPlugin.h"

#include <log.h>

#include <brayns/common/ActionInterface.h>
#include <brayns/engineapi/Camera.h>
#include <brayns/engineapi/Engine.h>
#include <brayns/engineapi/FrameBuffer.h>
#include <brayns/engineapi/Scene.h>
#include <brayns/parameters/ParametersManager.h>
#include <brayns/pluginapi/Plugin.h>

#include <fstream>

namespace mediamaker
{
using namespace brayns;

const std::string PLUGIN_VERSION = "0.3.1";
const std::string PLUGIN_API_PREFIX = "mm-";

// Number of floats used to define the camera
const size_t CAMERA_DEFINITION_SIZE = 12;

#define CATCH_STD_EXCEPTION()                  \
    catch (const std::runtime_error &e)        \
    {                                          \
        response.status = false;               \
        response.contents = e.what();          \
        PLUGIN_ERROR << e.what() << std::endl; \
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
        PLUGIN_INFO << "Registering '" + entryPoint + "' endpoint" << std::endl;
        actionInterface->registerRequest<Response>(entryPoint, [&]() {
            return _version();
        });

        entryPoint = PLUGIN_API_PREFIX + "set-odu-camera";
        PLUGIN_INFO << "Registering '" + entryPoint + "' endpoint" << std::endl;
        actionInterface->registerNotification<CameraDefinition>(
            entryPoint, [&](const CameraDefinition &s) { _setCamera(s); });

        entryPoint = PLUGIN_API_PREFIX + "get-odu-camera";
        PLUGIN_INFO << "Registering '" + entryPoint + "' endpoint" << std::endl;
        actionInterface->registerRequest<CameraDefinition>(
            entryPoint, [&]() -> CameraDefinition { return _getCamera(); });

        entryPoint = PLUGIN_API_PREFIX + "export-frames-to-disk";
        PLUGIN_INFO << "Registering '" + entryPoint + "' endpoint" << std::endl;
        actionInterface->registerNotification<ExportFramesToDisk>(
            entryPoint,
            [&](const ExportFramesToDisk &s) { _exportFramesToDisk(s); });

        entryPoint = PLUGIN_API_PREFIX + "get-export-frames-progress";
        PLUGIN_INFO << "Registering '" + entryPoint + "' endpoint" << std::endl;
        actionInterface->registerRequest<FrameExportProgress>(
            entryPoint, [&](void) -> FrameExportProgress {
                return _getFrameExportProgress();
            });
    }
}

Response MediaMakerPlugin::_version() const
{
    Response response;
    response.contents = PLUGIN_VERSION;
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
            _doExportFrameToDisk();

        if (_accumulationFrameNumber == _exportFramesToDiskPayload.spp)
        {
            _doExportFrameToDisk();
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
    brayns::Vector3f origin{o[0], o[1], o[2]};
    camera.setPosition(origin);

    // Target
    const auto &d = payload.direction;
    brayns::Vector3f direction{d[0], d[1], d[2]};
    camera.setTarget(origin + direction);

    // Up
    const auto &u = payload.up;
    brayns::Vector3f up{u[0], u[1], u[2]};

    // Orientation
    const glm::quat q = glm::inverse(
        glm::lookAt(origin, origin + direction,
                    up)); // Not quite sure why this should be inverted?!?
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

void MediaMakerPlugin::_doExportFrameToDisk()
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
        throw std::runtime_error("Unknown format: " +
                                 _exportFramesToDiskPayload.format);

    int flags = _exportFramesToDiskPayload.quality;
    if (fif == FIF_TIFF)
        flags = TIFF_NONE;

    brayns::freeimage::MemoryPtr memory(FreeImage_OpenMemory());

    FreeImage_SaveToMemory(fif, image.get(), memory.get(), flags);

    BYTE *pixels = nullptr;
    DWORD numPixels = 0;
    FreeImage_AcquireMemory(memory.get(), &pixels, &numPixels);

    std::string baseName = _baseName;
    if (baseName.empty())
    {
        char frame[7];
        sprintf(frame, "%05d", _frameNumber);
        baseName = frame;
    }
    const std::string filename = _exportFramesToDiskPayload.path + '/' +
                                 baseName + "." +
                                 _exportFramesToDiskPayload.format;

    std::ofstream file;
    file.open(filename, std::ios_base::binary);
    if (!file.is_open())
        PLUGIN_THROW(std::runtime_error("Failed to create " + filename));

    file.write((char *)pixels, numPixels);
    file.close();

    frameBuffer.clear();

    PLUGIN_INFO << "Frame saved to " << filename << std::endl;
}

void MediaMakerPlugin::_exportFramesToDisk(const ExportFramesToDisk &payload)
{
    _exportFramesToDiskPayload = payload;
    _exportFramesToDiskDirty = true;
    _frameNumber = payload.startFrame;
    _accumulationFrameNumber = 0;
    _baseName = payload.baseName;
    auto &frameBuffer = _api->getEngine().getFrameBuffer();
    frameBuffer.clear();
    const size_t nbFrames = _exportFramesToDiskPayload.endFrame -
                            _exportFramesToDiskPayload.startFrame;
    PLUGIN_INFO << "-----------------------------------------------------------"
                   "---------------------"
                << std::endl;
    PLUGIN_INFO << "Movie settings               :" << std::endl;
    PLUGIN_INFO << "- Samples per pixel          : " << payload.spp
                << std::endl;
    PLUGIN_INFO << "- Frame size                 : " << frameBuffer.getSize()
                << std::endl;
    PLUGIN_INFO << "- Export folder              : " << payload.path
                << std::endl;
    PLUGIN_INFO << "- Export intermediate frames : "
                << (payload.exportIntermediateFrames ? "Yes" : "No")
                << std::endl;
    PLUGIN_INFO << "- Start frame                : " << payload.startFrame
                << std::endl;
    PLUGIN_INFO << "- End frame                  : " << payload.endFrame
                << std::endl;
    PLUGIN_INFO << "- Frame base name            : " << payload.baseName
                << std::endl;
    PLUGIN_INFO << "- Number of frames           : " << nbFrames << std::endl;
    PLUGIN_INFO << "-----------------------------------------------------------"
                   "---------------------"
                << std::endl;
}

FrameExportProgress MediaMakerPlugin::_getFrameExportProgress()
{
    FrameExportProgress result;
    float percentage = 1.f;
    const size_t nbFrames =
        _exportFramesToDiskPayload.cameraInformation.size() /
        CAMERA_DEFINITION_SIZE;
    const size_t totalNumberOfFrames =
        nbFrames * _exportFramesToDiskPayload.spp;

    if (totalNumberOfFrames != 0)
    {
        const float currentProgress =
            _frameNumber * _exportFramesToDiskPayload.spp +
            _accumulationFrameNumber;
        percentage = currentProgress / float(totalNumberOfFrames);
    }
    result.progress = percentage;
    result.done = !_exportFramesToDiskDirty;
    PLUGIN_DEBUG << "Percentage = " << result.progress
                 << ", Done = " << (result.done ? "True" : "False")
                 << std::endl;
    return result;
}

extern "C" ExtensionPlugin *brayns_plugin_create(int /*argc*/, char ** /*argv*/)
{
    PLUGIN_INFO << "Initializing Media Maker plug-in (version "
                << PLUGIN_VERSION << ")" << std::endl;
    PLUGIN_INFO << std::endl;
    PLUGIN_INFO << "_|      _|                  _|  _|                _|      "
                   "_|            _|                          "
                << std::endl;
    PLUGIN_INFO << "_|_|  _|_|    _|_|      _|_|_|        _|_|_|      _|_|  "
                   "_|_|    _|_|_|  _|  _|      _|_|    _|  _|_|"
                << std::endl;
    PLUGIN_INFO << "_|  _|  _|  _|_|_|_|  _|    _|  _|  _|    _|      _|  _|  "
                   "_|  _|    _|  _|_|      _|_|_|_|  _|_|    "
                << std::endl;
    PLUGIN_INFO << "_|      _|  _|        _|    _|  _|  _|    _|      _|      "
                   "_|  _|    _|  _|  _|    _|        _|      "
                << std::endl;
    PLUGIN_INFO << "_|      _|    _|_|_|    _|_|_|  _|    _|_|_|      _|      "
                   "_|    _|_|_|  _|    _|    _|_|_|  _|      "
                << std::endl;
    PLUGIN_INFO << std::endl;
    return new MediaMakerPlugin();
}

} // namespace mediamaker
