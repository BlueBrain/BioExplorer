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

namespace bioexplorer
{
DBConnector::DBConnector(const std::string& connectionString,
                         const std::string& schema)
    : _connection(connectionString)
    , _schema(schema)
{
}

DBConnector::~DBConnector()
{
    _connection.disconnect();
}

void DBConnector::clearBricks()
{
    pqxx::work transaction(_connection);
    try
    {
        const auto sql = "DELETE FROM " + _schema + ".brick";
        PLUGIN_DEBUG << sql << std::endl;
        transaction.exec(sql);
        transaction.commit();
    }
    catch (pqxx::sql_error& e)
    {
        transaction.abort();
        PLUGIN_THROW(e);
    }
}

void DBConnector::insertBrick(const int32_t brickId, const uint32_t version,
                              const uint32_t nbModels,
                              const std::stringstream& buffer)
{
    pqxx::work transaction(_connection);
    try
    {
        const pqxx::binarystring tmp((void*)buffer.str().c_str(),
                                     buffer.str().size() * sizeof(char));
        transaction
            .parameterized("INSERT INTO " + _schema +
                           ".brick VALUES ($1, $2, $3, $4)")(brickId)(version)(
                nbModels)(tmp)
            .exec();
        transaction.commit();
        PLUGIN_DEBUG << "Brick ID " << brickId << " successfully inserted"
                     << std::endl;
    }
    catch (pqxx::sql_error& e)
    {
        transaction.abort();
        PLUGIN_THROW(e);
    }
}

std::stringstream DBConnector::selectBrick(const int32_t brickId,
                                           const uint32_t& version,
                                           uint32_t& nbModels)
{
    std::vector<char> tmp;
    pqxx::read_transaction transaction(_connection);
    try
    {
        const auto sql = "SELECT nb_models, buffer FROM " + _schema +
                         ".brick WHERE guid=" + std::to_string(brickId) +
                         " AND version=" + std::to_string(version);
        PLUGIN_DEBUG << sql << std::endl;
        auto res = transaction.exec(sql);
        for (auto c = res.begin(); c != res.end(); ++c)
        {
            nbModels = c[0].as<uint32_t>();
            const pqxx::binarystring buffer(c[1]);
            tmp.resize(buffer.size());
            memcpy(tmp.data(), buffer.data(), buffer.size());
        }
    }
    catch (pqxx::sql_error& e)
    {
        PLUGIN_THROW(e);
    }
    transaction.abort();

    std::stringstream s;
    if (nbModels > 0)
        std::copy(tmp.begin(), tmp.end(), std::ostream_iterator<char>(s));
    return s;
}

} // namespace bioexplorer
