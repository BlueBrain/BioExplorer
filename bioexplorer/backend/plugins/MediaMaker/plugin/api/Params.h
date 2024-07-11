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

#include <plugin/common/Types.h>

#include <bioexplorer/backend/science/common/Types.h>

#include <limits>
#include <string>
#include <vector>

namespace bioexplorer
{
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
    double apertureRadius{0.0};
    double focalDistance{1e6};
    double interpupillaryDistance{0.0};
};
bool from_json(CameraDefinition &param, const std::string &payload);
std::string to_json(const CameraDefinition &param);

struct ExportFramesToDisk
{
    std::string path;
    std::string baseName;
    std::string format;
    std::vector<double> size;
    uint16_t quality{100};
    uint16_t spp{0};
    uint16_t startFrame{0};
    uint16_t endFrame{std::numeric_limits<uint16_t>::max()};
    bool exportIntermediateFrames{false};
    uint64_ts animationInformation;
    doubles cameraInformation;
    FrameBufferMode frameBufferMode{FrameBufferMode::color};
    std::string keywords;
};
bool from_json(ExportFramesToDisk &param, const std::string &payload);

struct FrameExportProgress
{
    double progress;
    bool done;
};
std::string to_json(const FrameExportProgress &payload);

struct CameraHandlerDetails
{
    std::vector<double> origins;
    std::vector<double> directions;
    std::vector<double> ups;
    std::vector<double> apertureRadii;
    std::vector<double> focalDistances;
    uint16_t stepsBetweenKeyFrames;
    uint16_t numberOfSmoothingSteps;
};
std::string to_json(const CameraHandlerDetails &payload);
bool from_json(CameraHandlerDetails &param, const std::string &payload);

} // namespace mediamaker
} // namespace bioexplorer
