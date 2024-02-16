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
