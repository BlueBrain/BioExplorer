/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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
