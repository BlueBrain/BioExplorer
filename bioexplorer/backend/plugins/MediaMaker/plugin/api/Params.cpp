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

std::string to_json(const FrameExportProgress &exportProgress)
{
    try
    {
        nlohmann::json json;
        TO_JSON(exportProgress, json, progress);
        TO_JSON(exportProgress, json, done);
        return json.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

} // namespace mediamaker
} // namespace bioexplorer
