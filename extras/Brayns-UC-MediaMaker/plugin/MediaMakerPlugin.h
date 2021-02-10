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

#pragma once

#include <plugin/api/Params.h>

#include <brayns/pluginapi/ExtensionPlugin.h>

namespace mediamaker
{
/**
 * @brief This class implements the Media Maker plugin for Brayns
 */
class MediaMakerPlugin : public brayns::ExtensionPlugin
{
public:
    MediaMakerPlugin();

    void init() final;
    void preRender() final;
    void postRender() final;

private:
    Response _version() const;

    // Movie and frames
    ExportFramesToDisk _exportFramesToDiskPayload;
    bool _exportFramesToDiskDirty{false};
    uint16_t _frameNumber{0};
    int16_t _accumulationFrameNumber{0};

    void _setCamera(const CameraDefinition &);
    CameraDefinition _getCamera();
    void _exportFramesToDisk(const ExportFramesToDisk &payload);
    FrameExportProgress _getFrameExportProgress();
    void _doExportFrameToDisk();
};
} // namespace mediamaker
