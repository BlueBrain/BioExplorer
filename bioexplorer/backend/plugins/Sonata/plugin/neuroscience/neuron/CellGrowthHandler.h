/*
 * Copyright 2020-2024 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <platform/core/common/simulation/AbstractAnimationHandler.h>

#include <platform/core/common/Api.h>
#include <platform/core/common/Types.h>

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
/**
 * @brief The CellGrowthHandler class handles distance to the soma
 */
class CellGrowthHandler : public core::AbstractAnimationHandler
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

    core::AbstractSimulationHandlerPtr clone() const final;
};
using CellGrowthHandlerPtr = std::shared_ptr<CellGrowthHandler>;
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
