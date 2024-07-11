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

#include "Params.h"
#include <common/json.hpp>

namespace bioexplorer
{
namespace mediamaker
{
#ifndef PLATFORM_DEBUG_JSON_ENABLED
#define FROM_JSON(PARAM, JSON, NAME) PARAM.NAME = JSON[#NAME].get<decltype(PARAM.NAME)>()
#else
#define FROM_JSON(PARAM, JSON, NAME)                                                        \
    try                                                                                     \
    {                                                                                       \
        PARAM.NAME = JSON[#NAME].get<decltype(PARAM.NAME)>();                               \
    }                                                                                       \
    catch (...)                                                                             \
    {                                                                                       \
        PLUGIN_ERROR << "JSON parsing error for attribute '" << #NAME << "'!" << std::endl; \
        throw;                                                                              \
    }
#endif
#define TO_JSON(PARAM, JSON, NAME) JSON[#NAME] = PARAM.NAME

std::string to_json(const Response &param)
{
    try
    {
        nlohmann::json js;

        TO_JSON(param, js, status);
        TO_JSON(param, js, contents);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

// Movies and frames
bool from_json(CameraDefinition &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, origin);
        FROM_JSON(param, js, direction);
        FROM_JSON(param, js, up);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

std::string to_json(const CameraDefinition &param)
{
    try
    {
        nlohmann::json js;

        TO_JSON(param, js, origin);
        TO_JSON(param, js, direction);
        TO_JSON(param, js, up);
        TO_JSON(param, js, apertureRadius);
        TO_JSON(param, js, focalDistance);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

bool from_json(ExportFramesToDisk &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, path);
        FROM_JSON(param, js, baseName);
        FROM_JSON(param, js, format);
        FROM_JSON(param, js, size);
        FROM_JSON(param, js, quality);
        FROM_JSON(param, js, spp);
        FROM_JSON(param, js, startFrame);
        FROM_JSON(param, js, endFrame);
        FROM_JSON(param, js, exportIntermediateFrames);
        FROM_JSON(param, js, animationInformation);
        FROM_JSON(param, js, cameraInformation);
        FROM_JSON(param, js, frameBufferMode);
        FROM_JSON(param, js, keywords);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

std::string to_json(const FrameExportProgress &param)
{
    try
    {
        nlohmann::json json;
        TO_JSON(param, json, progress);
        TO_JSON(param, json, done);
        return json.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

std::string to_json(const CameraHandlerDetails &param)
{
    try
    {
        nlohmann::json json;
        TO_JSON(param, json, origins);
        TO_JSON(param, json, directions);
        TO_JSON(param, json, ups);
        TO_JSON(param, json, apertureRadii);
        TO_JSON(param, json, focalDistances);
        TO_JSON(param, json, stepsBetweenKeyFrames);
        TO_JSON(param, json, numberOfSmoothingSteps);
        return json.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

bool from_json(CameraHandlerDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, origins);
        FROM_JSON(param, js, directions);
        FROM_JSON(param, js, ups);
        FROM_JSON(param, js, apertureRadii);
        FROM_JSON(param, js, focalDistances);
        FROM_JSON(param, js, stepsBetweenKeyFrames);
        FROM_JSON(param, js, numberOfSmoothingSteps);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

} // namespace mediamaker
} // namespace bioexplorer
