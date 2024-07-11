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