/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#pragma once

#include <Defines.h>

#include <bioexplorer/core/plugin/common/Types.h>

using namespace brayns;

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
    std::string connectionString;
    std::string schema;
    size_t simulationId;
    int32_ts metaboliteIds;
    int32_t referenceFrame;
    bool relativeConcentration{false};
} AttachHandlerDetails;

typedef struct
{
    uint32_t guid;
    std::string code;
    Vector3f color;
} Location;
using Locations = std::vector<Location>;

} // namespace metabolism
} // namespace bioexplorer
