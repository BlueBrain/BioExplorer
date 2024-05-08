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

#include "DBConnector.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

#include <platform/core/common/geometry/TriangleMesh.h>

#include <omp.h>
#include <pqxx/pqxx>

using namespace core;

namespace bioexplorer
{
using namespace details;
using namespace common;
using namespace morphology;
using namespace molecularsystems;
using namespace vasculature;
using namespace connectomics;

namespace io
{
namespace db
{
const std::string DB_SCHEMA_OUT_OF_CORE = "outofcore";
const std::string DB_SCHEMA_METABOLISM = "metabolism";
const std::string DB_SCHEMA_CONNECTOME = "connectome";

DBConnector* DBConnector::_instance = nullptr;
std::mutex DBConnector::_mutex;

#define CHECK_DB_INITIALIZATION \
    if (!_initialized)          \
    PLUGIN_THROW("Database connection has not been initialized")

DBConnector::DBConnector() {}

DBConnector::~DBConnector()
{
    for (auto connection : _connections)
        if (connection)
            connection->disconnect();
}

void DBConnector::init(const CommandLineArguments& arguments)
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
        if (argument.first == ARG_DB_NB_CONNECTIONS)
            _dbNbConnections = std::stoi(argument.second.c_str());
        if (argument.first == ARG_DB_BATCH_SIZE)
            _dbBatchSize = std::stoi(argument.second.c_str());
    }

    _connectionString =
        "host=" + dbHost + " port=" + dbPort + " dbname=" + dbName + " user=" + dbUser + " password=" + dbPassword;

    PLUGIN_DB_INFO(1, _connectionString);

    for (size_t i = 0; i < _dbNbConnections; ++i)
    {
        try
        {
            _connections.push_back(ConnectionPtr(new pqxx::connection(_connectionString)));
        }
        catch (const pqxx::pqxx_exception& e)
        {
            PLUGIN_THROW(
                "Failed to connect to database, check command line "
                "parameters. " +
                std::string(e.base().what()));
        }
    }
    _initialized = true;
    PLUGIN_DB_INFO(1, "Initialized " << _dbNbConnections << " connections to database");
}

void DBConnector::clearBricks()
{
    pqxx::work transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        const auto sql = "DELETE FROM " + DB_SCHEMA_OUT_OF_CORE + ".brick";
        PLUGIN_DB_INFO(1, sql);
        transaction.exec(sql);
        transaction.commit();
        PLUGIN_DB_TIMER(chrono.elapsed(), "clearBricks()");
    }
    catch (pqxx::sql_error& e)
    {
        transaction.abort();
        PLUGIN_THROW(e.what());
    }
}

const OOCSceneConfigurationDetails DBConnector::getSceneConfiguration()
{
    CHECK_DB_INITIALIZATION
    OOCSceneConfigurationDetails sceneConfiguration;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        const auto sql =
            "SELECT scene_size_x, scene_size_y, scene_size_z, nb_bricks, "
            "description FROM " +
            DB_SCHEMA_OUT_OF_CORE + ".configuration";
        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            sceneConfiguration.sceneSize = Vector3d(c[0].as<double>(), c[1].as<double>(), c[2].as<double>());
            sceneConfiguration.nbBricks = c[3].as<uint32_t>();
            sceneConfiguration.description = c[4].as<std::string>();
            if (sceneConfiguration.nbBricks == 0)
                PLUGIN_THROW("Invalid number of bricks)");
            sceneConfiguration.brickSize =
                sceneConfiguration.sceneSize / static_cast<double>(sceneConfiguration.nbBricks);
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getSceneConfiguration()");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }
    return sceneConfiguration;
}

void DBConnector::insertBrick(const int32_t brickId, const uint32_t version, const uint32_t nbModels,
                              const std::stringstream& buffer)
{
    CHECK_DB_INITIALIZATION
    pqxx::work transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        const pqxx::binarystring tmp((void*)buffer.str().c_str(), buffer.str().size() * sizeof(char));
        transaction.exec_params("INSERT INTO " + DB_SCHEMA_OUT_OF_CORE + ".brick VALUES ($1, $2, $3, $4)", brickId,
                                version, nbModels, tmp);
        transaction.commit();
        PLUGIN_DB_INFO(1, "Brick ID " << brickId << " successfully inserted");
        PLUGIN_DB_TIMER(chrono.elapsed(), "insertBrick(brickId=" << brickId << ", version=" << version
                                                                 << ", nbModels=" << nbModels << ", <buffer>)");
    }
    catch (pqxx::sql_error& e)
    {
        transaction.abort();
        PLUGIN_THROW(e.what());
    }
}

std::stringstream DBConnector::getBrick(const int32_t brickId, const uint32_t& version, uint32_t& nbModels)
{
    CHECK_DB_INITIALIZATION
    std::stringstream s;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        const auto sql = "SELECT nb_models, buffer FROM " + DB_SCHEMA_OUT_OF_CORE +
                         ".brick WHERE guid=" + std::to_string(brickId) + " AND version=" + std::to_string(version);
        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            nbModels = c[0].as<uint32_t>();
            if (nbModels > 0)
            {
                const pqxx::binarystring buffer(c[1]);
                std::copy(buffer.begin(), buffer.end(), std::ostream_iterator<char>(s));
            }
        }
        PLUGIN_DB_TIMER(chrono.elapsed(),
                        "getBrick(brickId=" << brickId << ", version=" << version << ", nbModels=" << nbModels << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return s;
}

GeometryNodes DBConnector::getVasculatureNodes(const std::string& populationName, const std::string& sqlCondition,
                                               const std::string& limits) const
{
    CHECK_DB_INITIALIZATION
    GeometryNodes nodes;
    const auto threadNum = omp_get_thread_num();
    const auto connection = _connections[threadNum % _dbNbConnections];
    pqxx::nontransaction transaction(*connection);
    try
    {
        Timer chrono;
        std::string sql =
            "SELECT guid, x, y, z, radius, section_guid, sub_graph_guid, "
            "pair_guid, entry_node_guid, region_guid FROM " +
            populationName + ".node WHERE region_guid IS NOT NULL";
        if (!sqlCondition.empty())
            sql += " AND " + sqlCondition;
        sql += " ORDER BY section_guid, guid";

        if (!limits.empty())
            sql += " " + limits;

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            GeometryNode node;
            const uint64_t guid = c[0].as<uint64_t>();
            node.position = Vector3d(c[1].as<double>(), c[2].as<double>(), c[3].as<double>());
            node.radius = c[4].as<double>();
            node.sectionId = c[5].as<uint64_t>();
            node.graphId = c[6].as<uint64_t>();
            node.pairId = c[7].as<uint64_t>();
            node.entryNodeId = c[8].as<uint64_t>();
            node.regionId = c[9].as<uint64_t>();
            nodes[guid] = node;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getVasculatureNodes(populationName=" << populationName
                                                                                << ", sqlCondition=" << sqlCondition
                                                                                << ", limits=" << limits << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return nodes;
}

uint64_ts DBConnector::getVasculatureSections(const std::string& populationName, const std::string& sqlCondition)
{
    CHECK_DB_INITIALIZATION
    uint64_ts sectionIds;
    auto connection = _connections[omp_get_thread_num() % _dbNbConnections];
    pqxx::nontransaction transaction(*connection);
    try
    {
        Timer chrono;
        std::string sql =
            "SELECT distinct(section_guid) FROM " + populationName + ".node WHERE region_guid IS NOT NULL";

        if (!sqlCondition.empty())
            sql += " AND " + sqlCondition;

        PLUGIN_DB_INFO(1, sql);
        const pqxx::result res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
            sectionIds.push_back(c[0].as<uint64_t>());
        PLUGIN_DB_TIMER(chrono.elapsed(), "getVasculatureSections(populationName="
                                              << populationName << ", sqlCondition=" << sqlCondition << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return sectionIds;
}

uint64_t DBConnector::getVasculatureNbNodes(const std::string& populationName, const std::string& sqlCondition)
{
    CHECK_DB_INITIALIZATION
    uint64_t nbSections;
    auto connection = _connections[omp_get_thread_num() % _dbNbConnections];
    pqxx::nontransaction transaction(*connection);
    try
    {
        Timer chrono;
        std::string sql = "SELECT COUNT(guid) FROM " + populationName + ".node";
        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;
        const pqxx::result res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
            nbSections = c[0].as<uint64_t>();
        PLUGIN_DB_TIMER(chrono.elapsed(), "getVasculatureNbNodes(populationName=" << populationName << ", sqlCondition="
                                                                                  << sqlCondition << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return nbSections;
}

Vector2d DBConnector::getVasculatureRadiusRange(const std::string& populationName,
                                                const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    Vector2d range;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT min(radius), max(radius) FROM " + populationName + ".node";
        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            range.x = c[0].as<double>();
            range.y = c[1].as<double>();
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getVasculatureRadiusRange(populationName="
                                              << populationName << ", sqlCondition=" << sqlCondition << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return range;
}

GeometryEdges DBConnector::getVasculatureEdges(const std::string& populationName, const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    GeometryEdges edges;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT start_node_guid, end_node_guid FROM " + populationName + ".edge";
        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
            edges[c[0].as<uint64_t>()] = c[1].as<uint64_t>();
        PLUGIN_DB_TIMER(chrono.elapsed(), "getVasculatureEdges(populationName=" << populationName << ", sqlCondition="
                                                                                << sqlCondition << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return edges;
}

Bifurcations DBConnector::getVasculatureBifurcations(const std::string& populationName) const
{
    CHECK_DB_INITIALIZATION
    Bifurcations bifurcations;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT e.source_node_guid, e.target_node_guid FROM " + populationName +
                          ".vasculature AS v, " + populationName +
                          ".edge AS e WHERE "
                          "v.bifurcation_guid !=0 AND e.source_node_guid=v.node_guid";

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            const auto sourceNodeId = c[0].as<uint64_t>();
            const auto targetNodeId = c[0].as<uint64_t>();

            bifurcations[sourceNodeId].push_back(targetNodeId);
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getVasculatureBifurcations(populationName=" << populationName << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return bifurcations;
}

SimulationReport DBConnector::getSimulationReport(const std::string& populationName,
                                                  const int32_t simulationReportId) const
{
    CHECK_DB_INITIALIZATION
    SimulationReport simulationReport;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql =
            "SELECT type_guid, description, start_time, end_time, time_step, "
            "time_units, data_units, debug_mode, guids FROM " +
            populationName + ".report WHERE guid=" + std::to_string(simulationReportId);

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            simulationReport.type = static_cast<ReportType>(c[0].as<uint64_t>());
            simulationReport.description = c[1].as<std::string>();
            simulationReport.startTime = c[2].as<double>();
            simulationReport.endTime = c[3].as<double>();
            simulationReport.timeStep = c[4].as<double>();
            simulationReport.timeUnits = c[5].as<std::string>();
            simulationReport.dataUnits = c[6].as<std::string>();
            simulationReport.debugMode = c[7].as<bool>();
            const pqxx::binarystring bytea(c[8]);
            uint64_ts guids;
            guids.resize(bytea.size() / sizeof(uint64_t));
            memcpy(&guids[0], bytea.data(), bytea.size());
            for (uint64_t i = 0; i < guids.size(); ++i)
                simulationReport.guids[guids[i]] = i;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(),
                        "getSimulationReport(populationName=" << populationName
                                                              << ", simulationReportId=" << simulationReportId << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return simulationReport;
}

floats DBConnector::getVasculatureSimulationTimeSeries(const std::string& populationName,
                                                       const int32_t simulationReportId, const int32_t frame) const
{
    CHECK_DB_INITIALIZATION
    floats values;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT values FROM " + populationName +
                          ".simulation_time_series WHERE report_guid=" + std::to_string(simulationReportId) +
                          " AND frame_guid=" + std::to_string(frame);

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            const pqxx::binarystring bytea(c[0]);
            values.resize(bytea.size() / sizeof(float));
            memcpy(&values.data()[0], bytea.data(), bytea.size());
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getVasculatureSimulationTimeSeries(populationName="
                                              << populationName << ", simulationReportId=" << simulationReportId
                                              << ", frame=" << frame << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return values;
}

AstrocyteSomaMap DBConnector::getAstrocytes(const std::string& populationName, const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    AstrocyteSomaMap somas;

    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT guid, x, y, z, radius FROM " + populationName + ".node";
        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;
        sql += " ORDER BY guid";

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            AstrocyteSoma soma;
            soma.center = Vector3d(c[1].as<float>(), c[2].as<float>(), c[3].as<float>());
            soma.radius = c[4].as<float>() * 0.25;
            somas[c[0].as<uint64_t>()] = soma;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(),
                        "getAstrocytes(populationName=" << populationName << ", sqlCondition=" << sqlCondition << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return somas;
}

SectionMap DBConnector::getAstrocyteSections(const std::string& populationName, const int64_t astrocyteId,
                                             const bool connectedToSomaOnly) const
{
    CHECK_DB_INITIALIZATION
    SectionMap sections;

    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql =
            "SELECT section_guid, section_type_guid, section_parent_guid, "
            "points FROM " +
            populationName + ".section WHERE morphology_guid=" + std::to_string(astrocyteId);
        if (connectedToSomaOnly)
            sql += " AND section_parent_guid=-1";
        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            Section section;
            const auto sectionId = c[0].as<uint64_t>();
            section.type = c[1].as<uint64_t>();
            section.parentId = c[2].as<int64_t>();
            const pqxx::binarystring bytea(c[3]);
            section.points.resize(bytea.size() / sizeof(Vector4f));
            memcpy(&section.points.data()[0], bytea.data(), bytea.size());

            section.length = 0.0;
            for (uint64_t i = 0; i < section.points.size() - 1; ++i)
                section.length += length(Vector3f(section.points[i + 1]) - Vector3f(section.points[i]));
            sections[sectionId] = section;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getAstrocyteSections(populationName=" << populationName << ", astrocyteId="
                                                                                 << astrocyteId << astrocyteId << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return sections;
}

EndFootMap DBConnector::getAstrocyteEndFeet(const std::string& vasculaturePopulationName,
                                            const uint64_t astrocyteId) const
{
    CHECK_DB_INITIALIZATION
    EndFootMap endFeet;

    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql =
            "SELECT c.guid, n.x, n.y, n.z, n.radius, "
            "c.vasculature_section_guid, c.vasculature_segment_guid, "
            "c.endfoot_compartment_length, c.endfoot_compartment_diameter "
            "* "
            "0.5 FROM " +
            DB_SCHEMA_CONNECTOME + ".glio_vascular as c, " + vasculaturePopulationName +
            ".node as n WHERE c.vasculature_node_guid=n.guid AND "
            "c.astrocyte_guid=" +
            std::to_string(astrocyteId);

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            EndFoot endFoot;
            const auto endFootId = c[0].as<uint64_t>();
            endFoot.nodes.push_back(Vector4f(c[1].as<float>(), c[2].as<float>(), c[3].as<float>(), c[4].as<float>()));
            endFoot.vasculatureSectionId = c[5].as<uint64_t>();
            endFoot.vasculatureSegmentId = c[6].as<uint64_t>();
            endFoot.length = c[7].as<double>() / 2.0;
            endFoot.radius = c[8].as<double>() / 2.0;
            endFeet[endFootId] = endFoot;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getAstrocyteEndFeet(vasculaturePopulationName="
                                              << vasculaturePopulationName << ", astrocyteId=" << astrocyteId << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return endFeet;
}

TriangleMesh DBConnector::getAstrocyteMicroDomain(const std::string& populationName, const uint64_t astrocyteId) const
{
    CHECK_DB_INITIALIZATION
    TriangleMesh mesh;

    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT vertices, indices FROM " + populationName +
                          ".micro_domain WHERE guid=" + std::to_string(astrocyteId);

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            const pqxx::binarystring vertices(c[0]);
            mesh.vertices.resize(vertices.size() / sizeof(Vector3f));
            memcpy(&mesh.vertices.data()[0], vertices.data(), vertices.size());
            const pqxx::binarystring indices(c[1]);
            mesh.indices.resize(indices.size() / sizeof(Vector3ui));
            memcpy(&mesh.indices.data()[0], indices.data(), indices.size());
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getAstrocyteMicroDomain(populationName="
                                              << populationName << ", astrocyteId=" << astrocyteId << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return mesh;
}

uint64_t DBConnector::getNumberOfNeurons(const std::string& populationName, const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION

    uint64_t nbNeurons = 0;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql =
            "SELECT COUNT(guid) FROM " + populationName + ".node";

        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            nbNeurons = c[0].as<uint64_t>();
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getNumberOfNeurons(populationName=" << populationName << ", sqlCondition="
                                                                              << sqlCondition << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return nbNeurons;
}

NeuronSomaMap DBConnector::getNeurons(const std::string& populationName, const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    NeuronSomaMap somas;

    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql =
            "SELECT guid, x, y, z, rotation_x, rotation_y, rotation_z, "
            "rotation_w, electrical_type_guid, morphological_type_guid, "
            "morphology_guid FROM " +
            populationName + ".node";

        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;

        sql += " ORDER BY guid";

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            NeuronSoma soma;
            soma.position = Vector3d(c[1].as<float>(), c[2].as<float>(), c[3].as<float>());
            soma.rotation = Quaterniond(c[7].as<float>(), c[4].as<float>(), c[5].as<float>(), c[6].as<float>());
            soma.eType = c[8].as<uint64_t>();
            soma.mType = c[9].as<uint64_t>();
            soma.layer = 0; // TODO
            soma.morphologyId = c[10].as<uint64_t>();
            somas[c[0].as<uint64_t>()] = soma;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getNeuronSections(populationName=" << populationName << ", sqlCondition="
                                                                              << sqlCondition << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return somas;
}

SectionMap DBConnector::getNeuronSections(const std::string& populationName, const uint64_t neuronId,
                                          const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    SectionMap sections;

    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql =
            "SELECT s.section_guid, s.section_type_guid, "
            "s.section_parent_guid, s.points::bytea FROM " +
            populationName + ".node AS n, " + populationName +
            ".section AS s WHERE n.morphology_guid=s.morphology_guid "
            "AND n.guid=" +
            std::to_string(neuronId);
        if (!sqlCondition.empty())
            sql += " AND " + sqlCondition;
        sql += " ORDER BY s.section_guid";

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            Section section;
            const auto sectionId = c[0].as<uint64_t>();
            section.type = c[1].as<uint64_t>();
            section.parentId = c[2].as<int64_t>();
            const pqxx::binarystring bytea(c[3]);
            section.points.resize(bytea.size() / sizeof(Vector4f));
            memcpy(&section.points.data()[0], bytea.data(), bytea.size());

            section.length = 0.0;
            for (uint64_t i = 0; i < section.points.size() - 1; ++i)
                section.length += length(Vector3f(section.points[i + 1]) - Vector3f(section.points[i]));

            sections[sectionId] = section;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(),
                        "getNeuronSections(populationName=" << populationName << ", neuronId=" << neuronId
                                                            << ", sqlCondition=" << sqlCondition << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return sections;
}

SectionSynapseMap DBConnector::getNeuronAfferentSynapses(const std::string& populationName, const uint64_t neuronId,
                                                         const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    SectionSynapseMap sectionSynapseMap;

    uint64_t nbSynapses = 0;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql =
            "SELECT preSynaptic_section_guid, preSynaptic_segment_guid, "
            "preSynaptic_segment_distance, "
            "postSynaptic_neuron_guid, postSynaptic_section_guid, "
            "postSynaptic_segment_guid, "
            "postSynaptic_segment_distance "
            "FROM " +
            populationName + ".synapse WHERE presynaptic_neuron_guid=" + std::to_string(neuronId) +
            " ORDER BY preSynaptic_section_guid";

        if (!sqlCondition.empty())
            sql += " AND " + sqlCondition;

        PLUGIN_DB_INFO(1, sql);
        const auto res = transaction.exec(sql);

        uint64_t previousPreSynapticSectionId = 0;
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            const auto preSynapticSectionId = c[0].as<uint64_t>();
            const auto preSynapticSegmentId = c[1].as<uint64_t>();
            Synapse synapse;
            synapse.type = MorphologySynapseType::afferent;
            synapse.preSynapticSegmentDistance = c[2].as<double>();
            synapse.postSynapticNeuronId = c[3].as<uint64_t>();
            synapse.postSynapticSectionId = c[4].as<uint64_t>();
            synapse.postSynapticSegmentId = c[5].as<uint64_t>();
            synapse.postSynapticSegmentDistance = c[6].as<double>();
            sectionSynapseMap[preSynapticSectionId][preSynapticSegmentId].push_back(synapse);
            ++nbSynapses;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(),
                        "getNeuronSynapses(populationName=" << populationName << ", neuronId=" << neuronId
                                                            << ", sqlCondition=" << sqlCondition << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    PLUGIN_DB_INFO(1, "Neuron " + std::to_string(neuronId) + " has " + std::to_string(nbSynapses) + " synapses")
    return sectionSynapseMap;
}

SpikesMap DBConnector::getNeuronSpikeReportValues(const std::string& populationName, const uint64_t reportId,
                                                  const double startTime, const double endTime) const
{
    CHECK_DB_INITIALIZATION
    SpikesMap spikes;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT node_guid, max(timestamp) FROM " + populationName +
                          ".spike_report WHERE report_guid=" + std::to_string(reportId) +
                          " AND timestamp>=" + std::to_string(startTime) + " AND timestamp<" + std::to_string(endTime) +
                          " GROUP BY node_guid";
        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
            spikes[c[0].as<uint64_t>()] = c[1].as<float>();
        PLUGIN_DB_TIMER(chrono.elapsed(),
                        "getNeuronSpikeReportValues(populationName=" << populationName << ", reportId=" << reportId
                                                                     << ", endTime=" << endTime << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return spikes;
}

uint64_tm DBConnector::getSimulatedNodesGuids(const std::string& populationName, const uint64_t reportId) const
{
    CHECK_DB_INITIALIZATION
    std::map<uint64_t, uint64_t> guids;
    const uint64_t elementSize = sizeof(uint64_t);
    Timer chrono;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        const std::string sql = "SELECT node_guid FROM " + populationName +
                                ".simulated_node WHERE report_guid=" + std::to_string(reportId) + " ORDER BY node_guid";
        PLUGIN_DB_INFO(1, sql);
        const auto res = transaction.exec(sql);
        uint64_t count = 0;
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            guids[c[0].as<uint64_t>()] = count;
            ++count;
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }
    PLUGIN_DB_TIMER(chrono.elapsed(), guids.size() << " guids returned by getSimulatedNodesGuids(populationName="
                                                   << populationName << ", reportId=" << reportId << ")");
    return guids;
}

void DBConnector::getNeuronSomaReportValues(const std::string& populationName, const uint64_t reportId,
                                            const uint64_t frame, floats& values) const
{
    CHECK_DB_INITIALIZATION
    const uint64_t elementSize = sizeof(float);

    const uint64_t bufferOffset = 1; // First byte of bytea must be ignored

    Timer chrono;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        const std::string sql = "SELECT values1::bytea, values2::bytea FROM " + populationName +
                                ".soma_report WHERE report_guid=" + std::to_string(reportId) +
                                " AND frame=" + std::to_string(frame);
        PLUGIN_DB_INFO(1, sql);
        const auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            const pqxx::binarystring buffer1(c[0]);
            const uint64_t nbValues1 = buffer1.size() / elementSize;
            const pqxx::binarystring buffer2(c[0]);
            const uint64_t nbValues2 = buffer2.size() / elementSize;
            values.resize(nbValues1 + nbValues2);
            memcpy(&values[0], &buffer1[0], buffer1.size());
            memcpy(&values[nbValues1], &buffer2[0], buffer2.size());
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }
    PLUGIN_DB_TIMER(chrono.elapsed(),
                    "getNeuronSomaReportValues(populationName=" << populationName << ", reportId=" << reportId << ")");
}

uint64_ts DBConnector::getNeuronSectionCompartments(const std::string& populationName, const uint64_t reportId,
                                                    const uint64_t nodeId, const uint64_t sectionId) const
{
    CHECK_DB_INITIALIZATION
    uint64_ts compartments;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT compartment_guid FROM " + populationName +
                          ".compartment_report WHERE report_guid=" + std::to_string(reportId) +
                          " AND node_guid=" + std::to_string(nodeId) +
                          " AND section_guid=" + std::to_string(sectionId) + " ORDER BY compartment_guid";
        PLUGIN_DB_INFO(1, sql);
        const auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
            compartments.push_back(c[0].as<uint64_t>());
        PLUGIN_DB_TIMER(chrono.elapsed(), "getNeuronSectionCompartments(populationName="
                                              << populationName << ", nodeId=" << nodeId << ", reportId=" << reportId
                                              << ", sectionId=" << sectionId << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return compartments;
}

floats DBConnector::getNeuronCompartmentReportValues(const std::string& populationName, const uint64_t reportId,
                                                     const uint64_t frame) const
{
    CHECK_DB_INITIALIZATION
    floats values;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        const size_t elementSize = sizeof(float);
        const size_t offset = 1; // First byte of bytea must be ignored
        std::string sql = "SELECT SUBSTRING(values::bytea from " + std::to_string(offset + frame * elementSize) +
                          " for " + std::to_string(elementSize) + ") FROM " + populationName +
                          ".compartment_report WHERE report_guid=" + std::to_string(reportId) +
                          " ORDER BY compartment_guid";
        PLUGIN_DB_INFO(1, sql);
        const auto res = transaction.exec(sql);
        values.resize(res.size());
        uint64_t index = 0;
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            const pqxx::binarystring buffer(c[0]);
            memcpy(&values[index], buffer.data(), buffer.size());
            ++index;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getNeuronCompartmentReportValues(populationName="
                                              << populationName << ", reportId=" << reportId << ", frame=" << frame
                                              << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return values;
}

uint64_ts DBConnector::getAtlasRegions(const std::string& populationName, const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    uint64_ts regions;

    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT guid, code, description FROM " + populationName + ".region";

        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;

        sql += " ORDER BY guid";

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
            regions.push_back(c[0].as<uint64_t>());
        PLUGIN_DB_TIMER(chrono.elapsed(), "getAtlasRegions(sqlCondition=" << sqlCondition << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return regions;
}

CellMap DBConnector::getAtlasCells(const std::string& populationName, const uint64_t regionId, const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    CellMap cells;

    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql =
            "SELECT guid, x, y, z, rotation_x, rotation_y, rotation_z, "
            "rotation_w, cell_type_guid, electrical_type_guid, region_guid "
            "FROM " +
            populationName + ".cell WHERE region_guid=" + std::to_string(regionId);

        if (!sqlCondition.empty())
            sql += " AND " + sqlCondition;

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            Cell cell;
            cell.position = Vector3d(c[1].as<float>(), c[2].as<float>(), c[3].as<float>());
            cell.rotation = Quaterniond(c[7].as<float>(), c[6].as<float>(), c[5].as<float>(), c[4].as<float>());
            cell.type = c[8].as<uint64_t>();
            cell.eType = c[9].as<int64_t>();
            cell.region = c[10].as<uint64_t>();
            cells[c[0].as<uint64_t>()] = cell;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(),
                        "getAtlasCells(regionId=" << regionId << ", sqlCondition=" << sqlCondition << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return cells;
}

TriangleMesh DBConnector::getAtlasMesh(const std::string& populationName, const uint64_t regionId) const
{
    CHECK_DB_INITIALIZATION
    TriangleMesh mesh;

    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        const std::string sql = "SELECT vertices, indices, normals, colors FROM " + populationName +
                                ".mesh WHERE guid=" + std::to_string(regionId);

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            const pqxx::binarystring vertices(c[0]);
            mesh.vertices.resize(vertices.size() / sizeof(Vector3f));
            memcpy(&mesh.vertices.data()[0], vertices.data(), vertices.size());

            const pqxx::binarystring indices(c[1]);
            mesh.indices.resize(indices.size() / sizeof(Vector3ui));
            memcpy(&mesh.indices.data()[0], indices.data(), indices.size());

            const pqxx::binarystring normals(c[2]);
            mesh.normals.resize(normals.size() / sizeof(Vector3f));
            memcpy(&mesh.normals.data()[0], normals.data(), normals.size());

            const pqxx::binarystring colors(c[3]);
            mesh.colors.resize(colors.size() / sizeof(Vector3f));
            memcpy(&mesh.colors.data()[0], colors.data(), colors.size());
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getAtlasMesh(regionId=" << regionId << ")");
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return mesh;
}

WhiteMatterStreamlines DBConnector::getWhiteMatterStreamlines(const std::string& populationName,
                                                              const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    WhiteMatterStreamlines streamlines;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT points FROM " + populationName + ".streamline";
        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            Vector3fs points;
            const pqxx::binarystring buffer(c[0]);
            points.resize(buffer.size() / sizeof(Vector3f));
            memcpy(&points.data()[0], buffer.data(), buffer.size());
            streamlines.push_back(points);
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getWhiteMatterStreamlines(populationName="
                                              << populationName << ", sqlCondition=" << sqlCondition << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return streamlines;
}

SynapsesMap DBConnector::getSynapses(const std::string& populationName, const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    SynapsesMap synapses;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql =
            "SELECT guid, postsynaptic_neuron_guid, postsynaptic_section_guid, "
            "postsynaptic_segment_guid, presynaptic_segment_distance, "
            "postsynaptic_segment_distance, presynaptic_center_x_position, "
            "presynaptic_center_y_position, presynaptic_center_z_position, "
            "postsynaptic_center_x_position, postsynaptic_center_y_position, "
            "postsynaptic_center_z_position FROM " +
            populationName + ".synapse";
        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;
        sql += " ORDER BY guid";

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            Synapse synapse;
            synapse.postSynapticNeuronId = c[1].as<uint64_t>();
            synapse.postSynapticSectionId = c[2].as<uint64_t>();
            synapse.postSynapticSegmentId = c[3].as<uint64_t>();
            synapse.preSynapticSegmentDistance = c[4].as<float>();
            synapse.postSynapticSegmentDistance = c[5].as<float>();
            synapse.preSynapticSurfacePosition = {c[6].as<float>(), c[7].as<float>(), c[8].as<float>()};
            synapse.postSynapticSurfacePosition = {c[9].as<float>(), c[10].as<float>(), c[11].as<float>()};
            synapses[c[0].as<uint64_t>()] = synapse;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(),
                        "getSynapses(populationName=" << populationName << ", sqlCondition=" << sqlCondition << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return synapses;
}

Vector3ds DBConnector::getSynapseEfficacyPositions(const std::string& populationName,
                                                   const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    Vector3ds positions;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT synapse_guid, x, y, z FROM " + populationName + ".synapse_efficacy_report";
        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;
        sql += " ORDER BY synapse_guid";

        PLUGIN_DB_INFO(1, sql);
        auto res = transaction.exec(sql);
        positions.resize(res.size());
        uint64_t i = 0;
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            const Vector3d position(c[1].as<float>(), c[2].as<float>(), c[3].as<float>());
            positions[i] = position;
            ++i;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(), "getSynapseEfficacyPositions(populationName="
                                              << populationName << ", sqlCondition=" << sqlCondition << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return positions;
}

std::map<uint64_t, floats> DBConnector::getSynapseEfficacyReportValues(const std::string& populationName,
                                                                       const uint64_t frame,
                                                                       const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    std::map<uint64_t, floats> reportValues;
    const auto elementSize = sizeof(float);
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT values::bytea FROM " + populationName +
                          ".synapse_efficacy_report AS e LEFT OUTER JOIN " + populationName +
                          ".synapse_efficacy_simulated_synapse AS s ON e.synapse_guid=s.guid";
        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;
        sql += " ORDER BY synapse_guid";

        PLUGIN_DB_INFO(1, sql);
        const auto res = transaction.exec(sql);
        uint64_t i = 0;
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            const pqxx::binarystring buffer(c[0]);
            const uint64_t bufferSize = buffer.size();
            floats values;
            values.resize(bufferSize / elementSize);
            memcpy(&values[0], buffer.data(), bufferSize);
            reportValues[i] = values;
            ++i;
        }
        PLUGIN_DB_TIMER(chrono.elapsed(),
                        "getSynapseEfficacyReportValues(populationName=" << populationName << ", frame=" << frame
                                                                         << ", sqlCondition=" << sqlCondition << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }
    return reportValues;
}

uint64_tm DBConnector::getSynaptome(const std::string& populationName, const std::string& sqlCondition) const
{
    CHECK_DB_INITIALIZATION
    uint64_tm synaptome;
    pqxx::nontransaction transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
    try
    {
        Timer chrono;
        std::string sql = "SELECT presynaptic_node_guid, postsynaptic_node_guid FROM " + populationName + ".synaptome";
        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;

        PLUGIN_DB_INFO(1, sql);
        const auto res = transaction.exec(sql);
        uint64_t i = 0;
        for (auto c = res.begin(); c != res.end(); ++c)
            synaptome[c[0].as<uint64_t>()] = c[1].as<uint64_t>();
        PLUGIN_DB_TIMER(chrono.elapsed(),
                        "getSynaptome(populationName=" << populationName << ",  sqlCondition=" << sqlCondition << ")");
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }
    return synaptome;
}

} // namespace db
} // namespace io
} // namespace bioexplorer
