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

#include <core/brayns/common/simulation/AbstractSimulationHandler.h>

#include <core/brayns/common/Api.h>
#include <core/brayns/common/Types.h>

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
/**
 * @brief The CellGrowthHandler class handles distance to the soma
 */
class CellGrowthHandler : public brayns::AbstractSimulationHandler
{
public:
    /**
     * @brief Default constructor
     */
    CellGrowthHandler(const uint32_t nbFrames);
    CellGrowthHandler(const CellGrowthHandler& rhs);
    ~CellGrowthHandler();

    void* getFrameData(const uint32_t) final;

    bool isReady() const final { return true; }

    brayns::AbstractSimulationHandlerPtr clone() const final;
};
using CellGrowthHandlerPtr = std::shared_ptr<CellGrowthHandler>;
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
