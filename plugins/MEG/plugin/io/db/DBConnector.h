/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#pragma once

#include <pqxx/pqxx>

#include <brayns/common/types.h>

namespace fieldrenderer
{
struct Point
{
    float x, y, z, value;
};
typedef std::vector<Point> Points;

struct SimulationInformation
{
    uint32_t nbFrames;
    float endTime;
    double timeStep;
    std::string timeUnit;
};

class DBConnector
{
public:
    DBConnector(const std::string& connectionString, const std::string& schema, const bool useCompartments);
    ~DBConnector();

    Points getAllNeurons();
    Points getSpikingNeurons(const float timestamp, const float delta, const size_t density = 100);
    SimulationInformation getSimulationInformation();
    brayns::Boxf getCircuitBoundingBox();

private:
    pqxx::connection _connection;
    std::string _schema;
    bool _useCompartments;
};

typedef std::shared_ptr<DBConnector> DBConnectorPtr;

} // namespace fieldrenderer
