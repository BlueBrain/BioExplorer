/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "DBConnector.h"

#include <plugin/common/Logs.h>

namespace bioexplorer
{
namespace io
{
namespace db
{
const std::string DB_SCHEMA_VASCULATURE = "vasculature";
const std::string DB_SCHEMA_METABOLISM = "metabolism";
const std::string DB_SCHEMA_OUT_OF_CORE = "outofcore";

DBConnector* DBConnector::_instance = nullptr;
std::mutex DBConnector::_mutex;

DBConnector::DBConnector() {}

DBConnector::~DBConnector()
{
    if (_connection)
        _connection->disconnect();
}

DBConnector& DBConnector::getInstance()
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_instance)
        _instance = new DBConnector();
    return *_instance;
}

void DBConnector::init(const CommandLineArguments& arguments)
{
    try
    {
        std::string dbHost, dbPort, dbUser, dbPassword, dbName;
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
        }

        const std::string connectionString =
            "host=" + dbHost + " port=" + dbPort + " dbname=" + dbName +
            " user=" + dbUser + " password=" + dbPassword;
        PLUGIN_ERROR(connectionString);
        _connection = ConnectionPtr(new pqxx::connection(connectionString));
    }
    catch (const pqxx::pqxx_exception& e)
    {
        PLUGIN_THROW(
            "Failed to connect to database, check command line parameters. " +
            std::string(e.base().what()));
    }
}

void DBConnector::clearBricks()
{
    pqxx::work transaction(*_connection);
    try
    {
        const auto sql = "DELETE FROM " + DB_SCHEMA_OUT_OF_CORE + ".brick";
        PLUGIN_DEBUG(sql);
        transaction.exec(sql);
        transaction.commit();
    }
    catch (pqxx::sql_error& e)
    {
        transaction.abort();
        PLUGIN_THROW(e.what());
    }
}

const OOCSceneConfigurationDetails DBConnector::getSceneConfiguration()
{
    OOCSceneConfigurationDetails sceneConfiguration;
    pqxx::read_transaction transaction(*_connection);
    try
    {
        const auto sql =
            "SELECT scene_size_x, scene_size_y, scene_size_z, nb_bricks, "
            "description FROM " +
            DB_SCHEMA_OUT_OF_CORE + ".configuration";
        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            sceneConfiguration.sceneSize =
                Vector3d(c[0].as<double>(), c[1].as<double>(),
                         c[2].as<double>());
            sceneConfiguration.nbBricks = c[3].as<uint32_t>();
            sceneConfiguration.description = c[4].as<std::string>();
            if (sceneConfiguration.nbBricks == 0)
                PLUGIN_THROW("Invalid number of bricks)");
            sceneConfiguration.brickSize =
                sceneConfiguration.sceneSize /
                static_cast<double>(sceneConfiguration.nbBricks);
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }
    transaction.abort();
    return sceneConfiguration;
}

void DBConnector::insertBrick(const int32_t brickId, const uint32_t version,
                              const uint32_t nbModels,
                              const std::stringstream& buffer)
{
    pqxx::work transaction(*_connection);
    try
    {
        const pqxx::binarystring tmp((void*)buffer.str().c_str(),
                                     buffer.str().size() * sizeof(char));
        transaction.exec_params("INSERT INTO " + DB_SCHEMA_OUT_OF_CORE +
                                    ".brick VALUES ($1, $2, $3, $4)",
                                brickId, version, nbModels, tmp);
        transaction.commit();
        PLUGIN_DEBUG("Brick ID " << brickId << " successfully inserted");
    }
    catch (pqxx::sql_error& e)
    {
        transaction.abort();
        PLUGIN_THROW(e.what());
    }
}

std::stringstream DBConnector::getBrick(const int32_t brickId,
                                        const uint32_t& version,
                                        uint32_t& nbModels)
{
    std::stringstream s;
    pqxx::read_transaction transaction(*_connection);
    try
    {
        const auto sql = "SELECT nb_models, buffer FROM " +
                         DB_SCHEMA_OUT_OF_CORE +
                         ".brick WHERE guid=" + std::to_string(brickId) +
                         " AND version=" + std::to_string(version);
        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            nbModels = c[0].as<uint32_t>();
            if (nbModels > 0)
            {
                const pqxx::binarystring buffer(c[1]);
                std::copy(buffer.begin(), buffer.end(),
                          std::ostream_iterator<char>(s));
            }
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }
    transaction.abort();

    return s;
}

uint64_t DBConnector::getVasculaturePopulationId(
    const std::string& populationName) const
{
    uint64_t populationId;
    pqxx::read_transaction transaction(*_connection);
    try
    {
        std::string sql = "SELECT guid FROM " + DB_SCHEMA_VASCULATURE +
                          ".population WHERE name='" + populationName + "'";
        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
            populationId = c[0].as<uint64_t>();
    }
    catch (pqxx::sql_error& e)
    {
        transaction.abort();
        PLUGIN_THROW(e.what());
    }
    transaction.abort();
    return populationId;
}

GeometryNodes DBConnector::getVasculatureNodes(
    const std::string& populationName, const std::string& filter) const
{
    const auto populationId = getVasculaturePopulationId(populationName);

    GeometryNodes nodes;
    pqxx::read_transaction transaction(*_connection);
    try
    {
        std::string sql =
            "SELECT n.node_guid, n.x, n.y, n.z, n.radius, "
            "v.section_guid, v.type_guid, v.subgraph_guid, v.pair_guid, "
            "v.entry_node_guid FROM " +
            DB_SCHEMA_VASCULATURE + ".node as n, " + DB_SCHEMA_VASCULATURE +
            ".vasculature as v WHERE n.node_guid=v.node_guid AND "
            "n.population_guid=v.population_guid AND "
            "n.population_guid=" +
            std::to_string(populationId);
        if (!filter.empty())
            sql += "AND " + filter;
        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            GeometryNode node;
            node.position = Vector3d(c[1].as<double>(), c[2].as<double>(),
                                     c[3].as<double>());
            node.radius = c[4].as<double>();
            node.sectionId = c[5].as<uint64_t>();
            node.type = c[6].as<uint64_t>();
            node.graphId = c[7].as<uint64_t>();
            node.pairId = c[8].as<uint64_t>();
            node.entryNodeId = c[9].as<uint64_t>();
            nodes[c[0].as<uint64_t>()] = node;
        }
    }
    catch (const pqxx::sql_error& e)
    {
        transaction.abort();
        PLUGIN_THROW(e.what());
    }
    transaction.abort();

    return nodes;
}

GeometryEdges DBConnector::getVasculatureEdges(
    const std::string& populationName, const std::string& filter) const
{
    const auto populationId = getVasculaturePopulationId(populationName);

    GeometryEdges edges;
    pqxx::read_transaction transaction(*_connection);
    try
    {
        std::string sql =
            "SELECT source_node_guid, target_node_guid FROM " +
            DB_SCHEMA_VASCULATURE +
            ".edge WHERE population_guid=" + std::to_string(populationId);

        if (!filter.empty())
            sql += "AND " + filter;

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
            edges[c[0].as<uint64_t>()] = c[1].as<uint64_t>();
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }
    transaction.abort();

    return edges;
}

SimulationReport DBConnector::getVasculatureSimulationReport(
    const std::string& populationName, const int32_t simulationReportId) const
{
    const auto populationId = getVasculaturePopulationId(populationName);

    SimulationReport simulationReport;
    pqxx::read_transaction transaction(*_connection);
    try
    {
        std::string sql =
            "SELECT description, start_time, end_time, time_step, time_units, "
            "data_units FROM " +
            DB_SCHEMA_VASCULATURE +
            ".simulation_report WHERE simulation_report_guid=" +
            std::to_string(simulationReportId) +
            " AND population_guid=" + std::to_string(populationId);

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            simulationReport.description = c[0].as<std::string>();
            simulationReport.startTime = c[1].as<double>();
            simulationReport.endTime = c[2].as<double>();
            simulationReport.timeStep = c[3].as<double>();
            simulationReport.timeUnits = c[4].as<std::string>();
            simulationReport.dataUnits = c[5].as<std::string>();
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }
    transaction.abort();

    return simulationReport;
}

floats DBConnector::getVasculatureSimulationTimeSeries(
    const int32_t simulationReportId, const int32_t frame) const
{
    floats values;
    pqxx::read_transaction transaction(*_connection);
    try
    {
        std::string sql =
            "SELECT values FROM " + DB_SCHEMA_VASCULATURE +
            ".simulation_time_series WHERE simulation_report_guid=" +
            std::to_string(simulationReportId) +
            " AND frame_guid=" + std::to_string(frame);

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            const pqxx::binarystring bytea(c[0]);
            values.resize(bytea.size());
            memcpy(&values.data()[0], bytea.data(), bytea.size());
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }
    transaction.abort();

    return values;
}
} // namespace db
} // namespace io
} // namespace bioexplorer
