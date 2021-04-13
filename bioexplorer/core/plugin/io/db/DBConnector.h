/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#include <pqxx/pqxx>

#include <brayns/common/types.h>
#include <plugin/common/Types.h>

namespace bioexplorer
{
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
     * @brief Construct a new DBConnector object
     *
     * @param connectionString
     * @param schema
     */
    DBConnector(const std::string& connectionString, const std::string& schema);

    /**
     * @brief Destroy the DBConnector object
     *
     */
    ~DBConnector();

    /**
     * @brief Remove all bricks from the PostgreSQL database
     *
     */
    void clearBricks();

    /**
     * @brief Get the Scene configuration
     *
     * @return Configuration of the out-of-code read-only scene
     */
    const OOCSceneConfigurationDetails getSceneConfiguration();

    /**
     * @brief Get the Brick object
     *
     * @param brickId Identifier of the brick
     * @param version Version of the binary buffer with the contents of the
     * brick
     * @param nbModels The number of models contained in the brick
     * @return std::stringstream The binary buffer with the contents of the
     * brick
     */
    std::stringstream getBrick(const int32_t brickId, const uint32_t& version,
                               uint32_t& nbModels);

    /**
     * @brief Inserts a brick into the PostgreSQL database
     *
     * @param brickId Identifier of the brick
     * @param version Version of the binary buffer with the contents of the
     * brick
     * @param nbModels The number of models contained in the brick
     * @param buffer The binary buffer with the contents of the brick
     */
    void insertBrick(const int32_t brickId, const uint32_t version,
                     const uint32_t nbModels, const std::stringstream& buffer);

private:
    pqxx::connection _connection;
    std::string _schema;
};

typedef std::shared_ptr<DBConnector> DBConnectorPtr;

} // namespace bioexplorer
