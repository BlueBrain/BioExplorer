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

#pragma once

#include <plugin/api/Params.h>
#include <plugin/common/Types.h>

#include <platform/core/pluginapi/ExtensionPlugin.h>

namespace bioexplorer
{
namespace mediamaker
{
/**
 * @brief This class implements the Media Maker plugin for Core
 */
class MediaMakerPlugin : public core::ExtensionPlugin
{
public:
    MediaMakerPlugin();

    void init() final;
    void preRender() final;
    void postRender() final;

private:
    Response _version() const;

    void _createRenderers();
#ifdef USE_OPTIX6
    void _createOptiXRenderers();
#endif

    // Movie and frames
    void _setCamera(const CameraDefinition &);
    CameraDefinition _getCamera();

    ExportFramesToDisk _exportFramesToDiskPayload;
    bool _exportFramesToDiskDirty{false};
    uint16_t _frameNumber{0};
    core::Vector2ui _frameBufferSize;
    int16_t _accumulationFrameNumber{0};
    std::string _baseName;

    void _exportFramesToDisk(const ExportFramesToDisk &payload);
    FrameExportProgress _getFrameExportProgress();
    void _exportDepthBuffer() const;
    void _exportColorBuffer() const;
    void _exportFrameToDisk() const;

    void _attachCameraHandler(const CameraHandlerDetails &payload);
};
} // namespace mediamaker
} // namespace bioexplorer
