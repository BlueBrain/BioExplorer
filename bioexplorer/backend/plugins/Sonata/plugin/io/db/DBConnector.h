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

#include <common/Types.h>

#include <platform/core/common/Types.h>

#include <pqxx/pqxx>

#include <mutex>

const size_t DEFAULT_DB_NB_CONNECTIONS = 8;

namespace sonataexplorer
{
namespace io
{
namespace db
{
using ConnectionPtr = std::shared_ptr<pqxx::connection>;

/**
 * @brief The DBConnector class allows the BioExplorer to communicate with a
 * PostgreSQL database. The DBConnector requires the pqxx library to be found at
 * compilation time.
 *
 */
class DBConnector
{
public:
    /**
     * @brief Get the Instance object
     *
     * @return GeneralSettings* Pointer to the object
     */
    static DBConnector& getInstance()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!_instance)
            _instance = new DBConnector();
        return *_instance;
    }

    /**
     * @brief Connects to the database using the provided command line arguments
     *
     * @param arguments Command line arguments
     */
    void init(const CommandLineArguments& arguments);

    /**
     * @brief Get the number of connections to the database
     *
     * @return size_t Number of connections to the database
     */
    size_t getNbConnections() const { return _dbNbConnections; }

    /**
     * @brief Get the number of connections to the database
     *
     * @return size_t Number of connections to the database
     */
    void importCircuitMorphologies(const std::string& populationName, const std::string& source,
                                   const std::string& morphologyPath);

    static std::mutex _mutex;
    static DBConnector* _instance;

private:
    DBConnector();
    ~DBConnector();

    size_t _dbNbConnections{DEFAULT_DB_NB_CONNECTIONS};

    std::string _connectionString;

    std::vector<ConnectionPtr> _connections;
    bool _initialized{false};
};

} // namespace db
} // namespace io
} // namespace sonataexplorer
