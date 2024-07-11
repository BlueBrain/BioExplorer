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

using namespace bioexplorer;
using namespace metabolism;

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
