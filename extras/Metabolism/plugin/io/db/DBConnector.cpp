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

#include "DBConnector.h"

#include <plugin/common/Logs.h>

#include <fstream>

#define DEFAULT_NUM_FRAMES 1000

namespace bioexplorer
{
namespace metabolism
{
DBConnector::DBConnector(const CommandLineArguments& args)
{
    _parseArguments(args);
}

DBConnector::DBConnector(const AttachHandlerDetails& payload)
    : _connection(new pqxx::connection(payload.connectionString))
    , _dbSchema(payload.schema)
    , _simulationId(payload.simulationId)
{
}

DBConnector::~DBConnector()
{
    _connection->disconnect();
}

Locations DBConnector::getLocations()
{
    pqxx::read_transaction transaction(*_connection);
    Locations locations;
    try
    {
        const std::string sql = "SELECT guid, code, red, green, blue FROM " +
                                _dbSchema + ".location ORDER BY guid";

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            Location location;
            location.guid = c[0].as<uint32_t>();
            location.code = c[1].as<std::string>();
            location.color =
                Vector3f(c[2].as<float>(), c[3].as<float>(), c[4].as<float>());
            locations.push_back(location);
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_ERROR(e.what());
    }
    transaction.abort();
    return locations;
}

uint32_t DBConnector::getNbFrames()
{
    pqxx::read_transaction transaction(*_connection);
    uint32_t nbFrames = 0;
    try
    {
        const std::string sql =
            "SELECT nb_frames FROM " + _dbSchema +
            ".simulation WHERE guid=" + std::to_string(_simulationId);

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            nbFrames = c[0].as<uint32_t>();
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_ERROR(e.what());
    }
    transaction.abort();
    return nbFrames;
}

std::map<uint32_t, float> DBConnector::getConcentrations(const uint32_t frame,
                                                         const size_ts& ids)
{
    std::map<uint32_t, float> values;
    pqxx::read_transaction transaction(*_connection);
    try
    {
        std::string sql =
            "SELECT v.location_guid, c.concentration / v.base_value FROM " +
            _dbSchema + ".variable as v, " + _dbSchema +
            ".concentration AS c WHERE v.guid=c.variable_guid AND "
            "c.timestamp=" +
            std::to_string(frame) +
            " AND c.simulation_guid=" + std::to_string(_simulationId);

        if (!ids.empty())
        {
            std::string idsAsString = "";
            for (const auto id : ids)
            {
                if (!idsAsString.empty())
                    idsAsString += ",";
                idsAsString += std::to_string(id);
            }
            sql += " AND v.guid in (" + idsAsString + ")";
        }
        sql += " ORDER BY v.location_guid";

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
            values[c[0].as<uint32_t>()] = (c[1].as<float>());
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_ERROR(e.what());
    }
    transaction.abort();
    PLUGIN_DEBUG(values.size() << " values");
    return values;
}

void DBConnector::_parseArguments(const CommandLineArguments& arguments)
{
    std::string dbHost, dbPort, dbUser, dbPassword, dbName, dbConnectionString;
    for (const auto& argument : arguments)
    {
        if (argument.first == ARG_DB_HOST)
            dbHost = argument.second;
        if (argument.first == ARG_DB_PORT)
            dbPort = argument.second;
        if (argument.first == ARG_DB_USER)
            dbUser = argument.second;
        if (argument.first == ARG_DB_PASSWORD)
            dbPassword = argument.second;
        if (argument.first == ARG_DB_NAME)
            dbName = argument.second;
        if (argument.first == ARG_DB_SCHEMA)
            _dbSchema = argument.second;
    }
    // Sanity checks
    dbConnectionString = "host=" + dbHost + " port=" + dbPort +
                         " dbname=" + dbName + " user=" + dbUser +
                         " password=" + dbPassword;
    _connection = std::unique_ptr<pqxx::connection>(
        new pqxx::connection(dbConnectionString));
}
} // namespace metabolism
} // namespace bioexplorer
