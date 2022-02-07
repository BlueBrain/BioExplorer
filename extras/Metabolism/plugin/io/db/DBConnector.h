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

#include <pqxx/pqxx>

#include <plugin/common/Types.h>

namespace bioexplorer
{
namespace metabolism
{
class DBConnector
{
public:
    DBConnector(const CommandLineArguments& args);
    DBConnector(const AttachHandlerDetails& payload);
    ~DBConnector();

    uint32_t getNbFrames();
    Locations getLocations();
    std::map<uint32_t, float> getConcentrations(const uint32_t frame,
                                                const size_ts& ids);

private:
    void _parseArguments(const CommandLineArguments& arguments);

    std::unique_ptr<pqxx::connection> _connection;
    std::string _dbSchema;
    size_t _simulationId;
};
} // namespace metabolism
} // namespace bioexplorer
