/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue Brain Project / EPFL
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
#include <bioexplorer/core/json.hpp>

namespace bioexplorer
{
namespace metabolism
{
#ifndef BRAYNS_DEBUG_JSON_ENABLED
#define FROM_JSON(PARAM, JSON, NAME) \
    PARAM.NAME = JSON[#NAME].get<decltype(PARAM.NAME)>()
#else
#define FROM_JSON(PARAM, JSON, NAME)                                          \
    try                                                                       \
    {                                                                         \
        PARAM.NAME = JSON[#NAME].get<decltype(PARAM.NAME)>();                 \
    }                                                                         \
    catch (...)                                                               \
    {                                                                         \
        PLUGIN_ERROR << "JSON parsing error for attribute '" << #NAME << "'!" \
                     << std::endl;                                            \
        throw;                                                                \
    }
#endif
#define TO_JSON(PARAM, JSON, NAME) JSON[#NAME] = PARAM.NAME

bool from_json(AttachHandlerDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, connectionString);
        FROM_JSON(param, js, schema);
        FROM_JSON(param, js, simulationId);
        FROM_JSON(param, js, metaboliteIds);
        FROM_JSON(param, js, referenceFrame);
        FROM_JSON(param, js, relativeConcentration);
    }
    catch (...)
    {
        return false;
    }
    return true;
}
} // namespace metabolism
} // namespace bioexplorer
