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

#include <Defines.h>

#include <bioexplorer/backend/science/common/Types.h>

namespace bioexplorer
{
namespace metabolism
{
// Classes and typedefs
class DBConnector;
using DBConnectorPtr = std::shared_ptr<DBConnector>;

class MetabolismHandler;
using MetabolismHandlerPtr = std::shared_ptr<MetabolismHandler>;
using CommandLineArguments = std::map<std::string, std::string>;
using Concentrations = std::map<uint32_t, float>;

// Command line arguments
const std::string ARG_DB_HOST = "--db-host";
const std::string ARG_DB_PORT = "--db-port";
const std::string ARG_DB_NAME = "--db-dbname";
const std::string ARG_DB_USER = "--db-user";
const std::string ARG_DB_PASSWORD = "--db-password";
const std::string ARG_DB_SCHEMA = "--db-schema";

typedef struct
{
    uint32_t guid;
    std::string code;
    core::Vector3f color;
} Location;
using Locations = std::vector<Location>;

typedef struct
{
    std::string connectionString;
    std::string schema;
    size_t simulationId;
    int32_ts metaboliteIds;
    int32_t referenceFrame;
    bool relativeConcentration{false};
} AttachHandlerDetails;
} // namespace metabolism
} // namespace bioexplorer
