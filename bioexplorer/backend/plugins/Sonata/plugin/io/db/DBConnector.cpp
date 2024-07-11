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

#include "DBConnector.h"

#include <common/Logs.h>
#include <common/Types.h>
#include <common/Utils.h>

#include <platform/core/common/geometry/TriangleMesh.h>

#include <omp.h>
#include <pqxx/pqxx>

using namespace core;

namespace sonataexplorer
{
namespace io
{
namespace db
{
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
    }

    _connectionString =
        "host=" + dbHost + " port=" + dbPort + " dbname=" + dbName + " user=" + dbUser + " password=" + dbPassword;

    PLUGIN_INFO(_connectionString);

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
    PLUGIN_INFO("Initialized " << _dbNbConnections << " connections to database");
}

void DBConnector::importCircuitMorphologies(const std::string& populationName, const std::string& source,
                                            const std::string& morphologyPath)
{
    CHECK_DB_INITIALIZATION

    PLUGIN_INFO("- Brion Circuit " << source)
    const brion::BlueConfig blueConfiguration(source);
    PLUGIN_INFO("- Brain Circuit")
    const brain::Circuit circuit(blueConfiguration);
    PLUGIN_INFO("- GIDs")
    const auto gids = circuit.getGIDs();
    PLUGIN_INFO("- " << gids.size() << " URIs")
    const auto uris = circuit.getMorphologyURIs(gids);
    brain::neuron::SectionTypes sectionTypes;
    sectionTypes.push_back(brain::neuron::SectionType::soma);
    sectionTypes.push_back(brain::neuron::SectionType::axon);
    sectionTypes.push_back(brain::neuron::SectionType::dendrite);
    sectionTypes.push_back(brain::neuron::SectionType::apicalDendrite);

    uint64_t morphologyId = 0;
#pragma omp parallel for private(morphologyId) num_threads(_dbNbConnections)
    for (morphologyId = 0; morphologyId < uris.size(); ++morphologyId)
    {
        pqxx::work transaction(*_connections[omp_get_thread_num() % _dbNbConnections]);
        try
        {
            Timer chrono;
            const std::string path = morphologyPath + uris[morphologyId].getPath();
            const brion::URI source(path);
            const brain::neuron::Morphology morphology(source);
            const auto sections = morphology.getSections(sectionTypes);
            for (const auto& section : sections)
            {
                const Vector4fs& samples = section.getSamples();
                const pqxx::binarystring points((void*)samples.data(), samples.size() * sizeof(Vector4f));
                const int sectionId = section.getID();
                const int parentId = section.hasParent() ? section.getParent().getID() : -1;
                const int sectionType = static_cast<int>(section.getType());
                const Vector3f p = samples[0];
                transaction.exec_params("INSERT INTO " + populationName + ".section VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                                        morphologyId, sectionId, parentId, sectionType, points, p.x, p.y, p.z);
            }
            transaction.commit();
        }
        catch (const pqxx::sql_error& e)
        {
            transaction.abort();
            PLUGIN_THROW(e.what());
        }
        if (omp_get_thread_num() == 0)
        {
            const float progress = (1 + morphologyId) * omp_get_num_threads();
            PLUGIN_PROGRESS("- Loading cells", progress, uris.size());
        }
    }
}
} // namespace db
} // namespace io
} // namespace sonataexplorer
