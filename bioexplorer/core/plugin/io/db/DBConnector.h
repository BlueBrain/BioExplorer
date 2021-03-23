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
class DBConnector
{
public:
    DBConnector(const std::string& connectionString, const std::string& schema);
    ~DBConnector();

    void clearBricks();
    std::stringstream selectBrick(const int32_t brickId,
                                  const uint32_t& version, uint32_t& nbModels);
    void insertBrick(const int32_t brickId, const uint32_t version,
                     const uint32_t nbModels, const std::stringstream& buffer);

private:
    pqxx::connection _connection;
    std::string _schema;
};

typedef std::shared_ptr<DBConnector> DBConnectorPtr;

} // namespace bioexplorer
