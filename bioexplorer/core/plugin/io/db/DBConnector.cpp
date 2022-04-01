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
#include <plugin/common/Utils.h>

namespace bioexplorer
{
using namespace morphology;
namespace io
{
namespace db
{
const std::string DB_SCHEMA_OUT_OF_CORE = "outofcore";
const std::string DB_SCHEMA_VASCULATURE = "vasculature";
const std::string DB_SCHEMA_METABOLISM = "metabolism";
const std::string DB_SCHEMA_ASTROCYTES = "astrocytes";
const std::string DB_SCHEMA_CONNECTOME = "connectome";

DBConnector* DBConnector::_instance = nullptr;
std::mutex DBConnector::_mutex;

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
    }

    _connectionString = "host=" + dbHost + " port=" + dbPort +
                        " dbname=" + dbName + " user=" + dbUser +
                        " password=" + dbPassword;

    PLUGIN_DEBUG(_connectionString);

    const auto nbConnections = omp_get_max_threads();
    for (size_t i = 0; i < nbConnections; ++i)
    {
        try
        {
            _connections.push_back(
                ConnectionPtr(new pqxx::connection(_connectionString)));
        }
        catch (const pqxx::pqxx_exception& e)
        {
            PLUGIN_THROW(
                "Failed to connect to database, check command line "
                "parameters. " +
                std::string(e.base().what()));
        }
    }
    PLUGIN_INFO(1,
                "Initialized " << nbConnections << " connections to database");
}

void DBConnector::clearBricks()
{
    pqxx::work transaction(*_connections[omp_get_thread_num()]);
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
    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
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
    return sceneConfiguration;
}

void DBConnector::insertBrick(const int32_t brickId, const uint32_t version,
                              const uint32_t nbModels,
                              const std::stringstream& buffer)
{
    pqxx::work transaction(*_connections[omp_get_thread_num()]);
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
    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
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

    return s;
}

uint64_t DBConnector::getVasculaturePopulationId(
    const std::string& populationName) const
{
    uint64_t populationId;
    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
    try
    {
        std::string sql = "SELECT guid FROM " + DB_SCHEMA_VASCULATURE +
                          ".population WHERE name='" + populationName + "'";
        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        if (res.empty())
            PLUGIN_THROW("Population " + populationName +
                         " could not be found");
        for (auto c = res.begin(); c != res.end(); ++c)
            populationId = c[0].as<uint64_t>();
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }
    return populationId;
}

GeometryNodes DBConnector::getVasculatureNodes(
    const std::string& populationName, const std::string& filter) const
{
    const auto populationId = getVasculaturePopulationId(populationName);

    GeometryNodes nodes;
    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
    try
    {
        std::string sql =
            "SELECT guid, x, y, z, radius, section_guid, sub_graph_guid, "
            "pair_guid, entry_node_guid FROM " +
            DB_SCHEMA_VASCULATURE +
            ".node WHERE population_guid=" + std::to_string(populationId);
        if (!filter.empty())
            sql += " AND " + filter;
        sql += " ORDER BY guid";
        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            GeometryNode node;
            const uint64_t guid = c[0].as<uint64_t>();
            node.position = Vector3d(c[1].as<double>(), c[2].as<double>(),
                                     c[3].as<double>());
            node.radius = c[4].as<double>();
            node.sectionId = c[5].as<uint64_t>();
            node.graphId = c[6].as<uint64_t>();
            node.pairId = c[7].as<uint64_t>();
            node.entryNodeId = c[8].as<uint64_t>();
            nodes[guid] = node;
        }
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return nodes;
}

GeometryEdges DBConnector::getVasculatureEdges(
    const std::string& populationName, const std::string& filter) const
{
    const auto populationId = getVasculaturePopulationId(populationName);

    GeometryEdges edges;
    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
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

    return edges;
}

Bifurcations DBConnector::getVasculatureBifurcations(
    const std::string& populationName) const
{
    const auto populationId = getVasculaturePopulationId(populationName);

    Bifurcations bifurcations;
    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
    try
    {
        std::string sql =
            "SELECT e.source_node_guid, e.target_node_guid FROM "
            "vasculature.vasculature AS v, vasculature.edge AS e WHERE "
            "v.bifurcation_guid !=0 AND e.source_node_guid=v.node_guid AND "
            "v.population_guid=" +
            std::to_string(populationId);

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            const auto sourceNodeId = c[0].as<uint64_t>();
            const auto targetNodeId = c[0].as<uint64_t>();

            bifurcations[sourceNodeId].push_back(targetNodeId);
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return bifurcations;
}

SimulationReport DBConnector::getVasculatureSimulationReport(
    const std::string& populationName, const int32_t simulationReportId) const
{
    const auto populationId = getVasculaturePopulationId(populationName);

    SimulationReport simulationReport;
    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
    try
    {
        std::string sql =
            "SELECT description, start_time, end_time, time_step, "
            "time_units, "
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

    return simulationReport;
}

floats DBConnector::getVasculatureSimulationTimeSeries(
    const int32_t simulationReportId, const int32_t frame) const
{
    floats values;
    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
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
            values.resize(bytea.size() / sizeof(float));
            memcpy(&values.data()[0], bytea.data(), bytea.size());
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return values;
}

AstrocyteSomaMap DBConnector::getAstrocytes(
    const std::string& sqlCondition) const
{
    AstrocyteSomaMap somas;

    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
    try
    {
        std::string sql = "SELECT guid, x, y, z, radius FROM " +
                          DB_SCHEMA_ASTROCYTES + ".node";

        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;

        sql += " ORDER BY guid";

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            AstrocyteSoma soma;
            soma.center =
                Vector3d(c[1].as<float>(), c[2].as<float>(), c[3].as<float>());
            soma.radius = c[4].as<float>() * 0.25;
            somas[c[0].as<uint64_t>()] = soma;
        }
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return somas;
}

SectionMap DBConnector::getAstrocyteSections(const int64_t astrocyteId) const
{
    SectionMap sections;

    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
    try
    {
        std::string sql =
            "SELECT section_guid, section_type_guid, section_parent_guid, "
            "points FROM " +
            DB_SCHEMA_ASTROCYTES +
            ".section WHERE morphology_guid=" + std::to_string(astrocyteId);
        PLUGIN_DEBUG(sql);
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
            sections[sectionId] = section;
        }
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return sections;
}

EndFootMap DBConnector::getAstrocyteEndFeet(const uint64_t astrocyteId) const
{
    EndFootMap endFeet;

    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
    try
    {
        std::string sql =
            "SELECT c.guid, n.x, n.y, n.z, n.radius, "
            "c.vasculature_section_guid, c.vasculature_segment_guid, "
            "c.endfoot_compartment_length, c.endfoot_compartment_diameter "
            "* "
            "0.5 FROM " +
            DB_SCHEMA_CONNECTOME + ".glio_vascular as c, " +
            DB_SCHEMA_VASCULATURE +
            ".node as n WHERE c.vasculature_node_guid=n.guid AND "
            "c.astrocyte_guid=" +
            std::to_string(astrocyteId);

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            EndFoot endFoot;
            const auto endFootId = c[0].as<uint64_t>();
            endFoot.nodes.push_back(Vector4f(c[1].as<float>(), c[2].as<float>(),
                                             c[3].as<float>(),
                                             c[4].as<float>()));
            endFoot.vasculatureSectionId = c[5].as<uint64_t>();
            endFoot.vasculatureSegmentId = c[6].as<uint64_t>();
            endFoot.length = c[7].as<double>();
            endFoot.radius = c[8].as<double>();
            endFeet[endFootId] = endFoot;
        }
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return endFeet;
}

NeuronSomaMap DBConnector::getNeurons(const std::string& populationName,
                                      const std::string& sqlCondition) const
{
    NeuronSomaMap somas;

    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
    try
    {
        std::string sql =
            "SELECT guid, x, y, z, rotation_x, rotation_y, rotation_z, "
            "rotation_w, electrical_type_guid, morphological_type_guid "
            "FROM " +
            populationName + ".node";

        if (!sqlCondition.empty())
            sql += " WHERE " + sqlCondition;

        sql += " ORDER BY guid";

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            NeuronSoma soma;
            soma.position =
                Vector3d(c[1].as<float>(), c[2].as<float>(), c[3].as<float>());
            soma.rotation = Quaterniond(c[7].as<float>(), c[4].as<float>(),
                                        c[5].as<float>(), c[6].as<float>());
            soma.eType = c[8].as<uint64_t>();
            soma.mType = c[9].as<uint64_t>();
            soma.layer = 0; // TODO
            somas[c[0].as<uint64_t>()] = soma;
        }
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return somas;
}

SectionMap DBConnector::getNeuronSections(const std::string& populationName,
                                          const uint64_t neuronId,
                                          const std::string& sqlCondition) const
{
    SectionMap sections;

    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
    try
    {
        std::string sql =
            "SELECT s.section_guid, s.section_type_guid, "
            "s.section_parent_guid, s.points FROM " +
            populationName + ".node as n, " + populationName +
            ".section as s WHERE n.morphology_guid=s.morphology_guid "
            "AND n.guid=" +
            std::to_string(neuronId);
        if (!sqlCondition.empty())
            sql += " AND " + sqlCondition;
        sql += " ORDER BY s.section_guid";

        PLUGIN_DEBUG(sql);
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
            sections[sectionId] = section;
        }
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return sections;
}

SynapseMap DBConnector::getNeuronSynapses(const std::string& populationName,
                                          const uint64_t neuronId,
                                          const std::string& sqlCondition) const
{
    SynapseMap synapses;

    pqxx::read_transaction transaction(*_connections[omp_get_thread_num()]);
    try
    {
        std::string sql =
            "SELECT guid, postsynaptic_neuron_guid, surface_x_position, "
            "surface_y_position, surface_z_position, center_x_position, "
            "center_y_position, center_z_position FROM " +
            populationName + ".synapse WHERE presynaptic_neuron_guid=" +
            std::to_string(neuronId);

        if (!sqlCondition.empty())
            sql += " AND " + sqlCondition;
        sql += " ORDER BY presynaptic_neuron_guid";

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            Synapse synapse;
            const auto synapseId = c[0].as<uint64_t>();
            synapse.preSynapticNeuron = neuronId;
            synapse.preSynapticNeuron = c[1].as<uint64_t>();
            synapse.surfacePosition =
                Vector3d(c[2].as<double>(), c[3].as<double>(),
                         c[4].as<double>());
            synapse.centerPosition =
                Vector3d(c[5].as<double>(), c[6].as<double>(),
                         c[7].as<double>());
            synapses[synapseId] = synapse;
        }
    }
    catch (const pqxx::sql_error& e)
    {
        PLUGIN_THROW(e.what());
    }

    return synapses;
}
} // namespace db
} // namespace io
} // namespace bioexplorer
