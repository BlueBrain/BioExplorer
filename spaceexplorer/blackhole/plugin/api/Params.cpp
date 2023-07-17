/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "Params.h"

#include <bioexplorer/backend/json.hpp>

namespace spaceexplorer
{
namespace blackhole
{
#ifndef BRAYNS_DEBUG_JSON_ENABLED
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

} // namespace blackhole
} // namespace spaceexplorer