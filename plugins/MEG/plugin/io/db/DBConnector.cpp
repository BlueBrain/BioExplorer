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

#include "DBConnector.h"

#include <plugin/common/Logs.h>

#include <fstream>

#define DEFAULT_NUM_FRAMES 1000

namespace fieldrenderer
{
DBConnector::DBConnector(const std::string& connectionString, const std::string& schema, const bool useCompartments)
    : _connection(connectionString)
    , _schema(schema)
    , _useCompartments(useCompartments)
{
}

DBConnector::~DBConnector()
{
    _connection.disconnect();
}

Points DBConnector::getAllNeurons()
{
    Points points;
    pqxx::read_transaction transaction(_connection);
    try
    {
        if (_useCompartments)
        {
            const std::string sql = "SELECT compartments FROM " + _schema + ".cell_compartments";
            PLUGIN_DEBUG(sql);
            auto res = transaction.exec(sql);
            for (auto c = res.begin(); c != res.end(); ++c)
            {
                const pqxx::binarystring positions(c[0]);
                for (uint64_t i = 0; i < positions.size() / sizeof(brayns::Vector4f); ++i)
                {
                    brayns::Vector4f pos;
                    memcpy(&pos, &positions[i * sizeof(brayns::Vector4f)], sizeof(brayns::Vector4f));
                    points.push_back({pos.x, pos.y, pos.z, pos.w});
                }
            }
        }
        else
        {
            const std::string sql = "SELECT p.x, p.y, p.z FROM " + _schema + ".cell_position AS p";

            PLUGIN_DEBUG(sql);
            auto res = transaction.exec(sql);
            for (auto c = res.begin(); c != res.end(); ++c)
                points.push_back({c[0].as<float>(), c[1].as<float>(), c[2].as<float>(), 0.f});
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_ERROR(e.what());
    }
    transaction.abort();
    PLUGIN_DEBUG(points.size() << " points");
    return points;
}

Points DBConnector::getSpikingNeurons(const float timestamp, const float delta, const size_t density)
{
    const float restVoltage = -65.0255f;
    Points points;
    pqxx::read_transaction transaction(_connection);
    try
    {
        if (_useCompartments)
        {
            const std::string sql =
                "SELECT cc.compartments, cc.nb_compartments, "
                "substring(sc.voltages, 1 + " +
                std::to_string(int(timestamp)) + " * 4 * cc.nb_compartments, 4 * cc.nb_compartments) FROM " + _schema +
                ".simulation_compartments AS sc, " + _schema + ".cell_compartments AS cc WHERE cc.guid=sc.guid";
            PLUGIN_INFO(sql);
            auto res = transaction.exec(sql);
            float minValue = std::numeric_limits<float>::max();
            float maxValue = -std::numeric_limits<float>::max();
            for (auto c = res.begin(); c != res.end(); ++c)
            {
                const pqxx::binarystring positions(c[0]);
                const uint64_t nbCompartments = c[1].as<uint64_t>();
                const pqxx::binarystring voltages(c[2]);

                std::vector<brayns::Vector4f> positionsBuffer;
                positionsBuffer.reserve(nbCompartments);
                memcpy(positionsBuffer.data(), &positions[0], nbCompartments * sizeof(brayns::Vector4f));

                std::vector<float> voltagesBuffer;
                voltagesBuffer.reserve(nbCompartments);
                memcpy(voltagesBuffer.data(), &voltages[0], nbCompartments * sizeof(float));

                for (uint64_t i = 0; i < nbCompartments; ++i)
                {
                    const brayns::Vector4f& position = positionsBuffer[i];
                    const float voltage = voltagesBuffer[i];
                    const float value = (voltage - restVoltage) / -restVoltage;
                    points.push_back({position.x, position.y, position.z, value});
                    minValue = std::min(minValue, voltage);
                    maxValue = std::max(maxValue, voltage);
                }
            }
            PLUGIN_INFO("Timestamp = " << timestamp << ", Range = [" << minValue << ", " << maxValue
                                       << "], Points = " << points.size());
        }
        else
        {
            std::string sql = "SELECT p.x, p.y, p.z, s.timestamp FROM " + _schema + ".cell_position AS p, " + _schema +
                              ".spikes_report AS s WHERE "
                              "s.guid=p.guid AND s.timestamp > " +
                              std::to_string(timestamp - delta) + " AND s.timestamp <= " + std::to_string(timestamp);
            size_t modulo = 1;
            if (density != 100)
            {
                modulo = 1.f / float(density) * 100;
                sql = sql + " AND s.guid % " + std::to_string(modulo) + " = 0";
            }

            PLUGIN_DEBUG(sql);
            auto res = transaction.exec(sql);
            for (auto c = res.begin(); c != res.end(); ++c)
            {
                const float intensity = std::max(0.f, -restVoltage * (1.f - (timestamp - c[3].as<float>()) / delta));
                points.push_back({c[0].as<float>(), c[1].as<float>(), c[2].as<float>(), intensity});
            }
            PLUGIN_INFO("Timestamp = " << timestamp << ", Density = " << density << ", Modulo = " << modulo
                                       << ", Points = " << points.size());
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_ERROR(e.what());
    }
    transaction.abort();
    PLUGIN_DEBUG(points.size() << " points");
    return points;
}

SimulationInformation DBConnector::getSimulationInformation()
{
    pqxx::read_transaction transaction(_connection);
    SimulationInformation si;
    try
    {
        std::string sql;
        if (_useCompartments)
            sql =
                "SELECT num_frames, end_time, time_step, time_unit FROM " + _schema + ".simulation_report WHERE guid=0";
        else
            sql = "SELECT " + std::to_string(DEFAULT_NUM_FRAMES) + ", max(timestamp), max(timestamp) / " +
                  std::to_string(DEFAULT_NUM_FRAMES) + ", 'ms' FROM " + _schema + ".spikes_report";

        PLUGIN_DEBUG(sql);
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            si.nbFrames = c[0].as<uint64_t>();
            si.endTime = c[1].as<float>();
            si.timeStep = c[2].as<float>();
            si.timeUnit = c[3].as<std::string>();
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_ERROR(e.what());
    }
    transaction.abort();
    PLUGIN_INFO("------------------------------");
    PLUGIN_INFO("Simulation information");
    PLUGIN_INFO("- Number of frames: " << si.nbFrames);
    PLUGIN_INFO("- End time        : " << si.endTime);
    PLUGIN_INFO("- Time step       : " << si.timeStep);
    PLUGIN_INFO("- Time unit       : " << si.timeUnit);
    PLUGIN_INFO("------------------------------");
    return si;
}

brayns::Boxf DBConnector::getCircuitBoundingBox()
{
    brayns::Boxf box;
    pqxx::read_transaction transaction(_connection);
    uint64_t nbFrames;
    try
    {
        if (_useCompartments)
        {
            const std::string sql = "SELECT compartments FROM " + _schema + ".cell_compartments";
            PLUGIN_DEBUG(sql);
            auto res = transaction.exec(sql);
            for (auto c = res.begin(); c != res.end(); ++c)
            {
                const pqxx::binarystring positions(c[0]);
                for (uint64_t i = 0; i < positions.size() / sizeof(brayns::Vector3f); ++i)
                {
                    brayns::Vector3f pos;
                    memcpy(&pos, &positions[i * sizeof(brayns::Vector3f)], sizeof(brayns::Vector3f));
                    box.merge({pos.x, pos.y, pos.z});
                }
            }
        }
        else
        {
            const std::string sql =
                "SELECT min(x), min(y), min(z), max(x), max(y), max(z) FROM " + _schema + ".cell_position";

            PLUGIN_DEBUG(sql);
            auto res = transaction.exec(sql);
            for (auto c = res.begin(); c != res.end(); ++c)
            {
                box.merge({c[0].as<float>(), c[1].as<float>(), c[2].as<float>()});
                box.merge({c[3].as<float>(), c[4].as<float>(), c[5].as<float>()});
            }
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_ERROR(e.what());
    }
    transaction.abort();
    PLUGIN_DEBUG("Bounding box: " << box);
    return box;
}
} // namespace fieldrenderer
