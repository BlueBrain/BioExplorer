/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include <plugin/api/Params.h>
#include <plugin/common/Types.h>
#include <plugin/io/db/DBConnector.h>

#include <brayns/api.h>
#include <brayns/common/simulation/AbstractSimulationHandler.h>

namespace bioexplorer
{
namespace metabolism
{
using namespace brayns;

/**
 * @brief The MetabolismHandler class handles metabolite concentrations
 */
class MetabolismHandler : public brayns::AbstractSimulationHandler
{
public:
    /**
     * @brief Default constructor
     */
    MetabolismHandler();
    MetabolismHandler(const CommandLineArguments& args);
    MetabolismHandler(const AttachHandlerDetails& payload);

    MetabolismHandler(const MetabolismHandler& rhs);
    ~MetabolismHandler();

    void* getFrameData(const uint32_t) final;

    bool isReady() const final { return true; }

    AbstractSimulationHandlerPtr clone() const final;

    void setMetaboliteIds(const int32_ts& metaboliteIds)
    {
        _metaboliteIds = metaboliteIds;
    }

private:
    DBConnectorPtr _connector{nullptr};

    int32_ts _metaboliteIds;
    Locations _locations;
    bool _relativeConcentration{false};
    uint32_t _referenceFrame{0};
};
} // namespace metabolism
} // namespace bioexplorer
