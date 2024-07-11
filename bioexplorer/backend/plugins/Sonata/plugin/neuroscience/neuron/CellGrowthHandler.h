/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
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
