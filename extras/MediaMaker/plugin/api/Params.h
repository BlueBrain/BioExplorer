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

#include <limits>
#include <string>
#include <vector>

namespace mediamaker
{
// Response
struct Response
{
    bool status{true};
    std::string contents;
};
std::string to_json(const Response &param);

// Movies and frames
struct CameraDefinition
{
    std::vector<double> origin;
    std::vector<double> direction;
    std::vector<double> up;
    double apertureRadius;
    double focusDistance;
    double interpupillaryDistance;
};
bool from_json(CameraDefinition &param, const std::string &payload);
std::string to_json(const CameraDefinition &param);

struct ExportFramesToDisk
{
    std::string path;
    std::string baseName;
    std::string format;
    uint16_t quality{100};
    uint16_t spp{0};
    uint16_t startFrame{0};
    uint16_t endFrame{std::numeric_limits<uint16_t>::max()};
    bool exportIntermediateFrames{false};
    std::vector<uint64_t> animationInformation;
    std::vector<double> cameraInformation;
};
bool from_json(ExportFramesToDisk &param, const std::string &payload);

struct FrameExportProgress
{
    float progress;
    bool done;
};
std::string to_json(const FrameExportProgress &exportProgress);

} // namespace mediamaker
