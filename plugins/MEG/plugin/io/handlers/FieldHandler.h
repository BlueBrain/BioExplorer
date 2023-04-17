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

#include <plugin/api/Params.h>
#include <plugin/io/db/DBConnector.h>

#include <brayns/api.h>
#include <brayns/common/simulation/AbstractSimulationHandler.h>
#include <brayns/common/types.h>

namespace fieldrenderer
{
using namespace brayns;

struct CellInfo
{
    Vector3f position;
    floats voltages;
};

/**
 * @brief The FieldHandler class handles electromagnetic fields data structures
 */
class FieldHandler : public brayns::AbstractSimulationHandler
{
public:
    /**
     * @brief Default constructor
     */
    FieldHandler();
    FieldHandler(const std::string& uri, const std::string& schema, const bool useCompartments);
    FieldHandler(const FieldHandler& rhs);
    ~FieldHandler();

    void* getFrameData(const uint32_t) final;
    Boxf getCircuitBoundingBox() const { return _boundingBox; };
    void setParams(const SimulationParameters& params)
    {
        _params = params;
        _nbFrames = params.nbFrames;
        _dt = _duration / _nbFrames;
    }

    bool isReady() const final { return true; }

    AbstractSimulationHandlerPtr clone() const final;

    void importFromFile(const std::string& filename);

private:
    std::vector<CellInfo> _cells;
    size_t _nbFloatsPerElement{4};

    SimulationParameters _params;
    float _duration{0.f};

    DBConnectorPtr _connector{nullptr};
    Boxf _boundingBox;
};
typedef std::shared_ptr<FieldHandler> FieldHandlerPtr;
} // namespace fieldrenderer
