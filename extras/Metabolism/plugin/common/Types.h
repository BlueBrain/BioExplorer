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
typedef std::shared_ptr<DBConnector> DBConnectorPtr;

class MetabolismHandler;
typedef std::shared_ptr<MetabolismHandler> MetabolismHandlerPtr;

typedef std::map<std::string, std::string> CommandLineArguments;

// Command line arguments
const std::string ARG_DB_HOST = "--db-host";
const std::string ARG_DB_PORT = "--db-port";
const std::string ARG_DB_NAME = "--db-dbname";
const std::string ARG_DB_USER = "--db-user";
const std::string ARG_DB_PASSWORD = "--db-password";
const std::string ARG_DB_SCHEMA = "--db-schema";

struct AttachHandlerDetails
{
    std::string connectionString;
    std::string schema;
    size_t simulationId;
    int32_ts metaboliteIds;
    bool relativeConcentration{false};
    double scale{1.0};
};

struct Location
{
    uint32_t guid;
    std::string code;
    Vector3f color;
};
typedef std::vector<Location> Locations;

} // namespace metabolism
} // namespace bioexplorer
