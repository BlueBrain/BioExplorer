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

#include <science/api/Params.h>

#include <platform/core/common/Api.h>
#include <platform/core/common/Types.h>
#include <platform/core/common/simulation/AbstractAnimationHandler.h>

namespace bioexplorer
{
namespace vasculature
{
/**
 * @brief The VasculatureHandler class handles the mapping of the vasculature
 * simulation to the geometry
 */
class VasculatureHandler : public core::AbstractAnimationHandler
{
public:
    /**
     * @brief Default constructor
     */
    VasculatureHandler(const details::VasculatureReportDetails& details);

    /**
     * @copydoc core::AbstractAnimationHandler::getFrameData
     */
    void* getFrameData(const uint32_t) final;

    /**
     * @copydoc core::AbstractAnimationHandler::isReady
     */
    bool isReady() const final { return true; }

    /**
     * @copydoc core::AbstractAnimationHandler::clone
     */
    core::AbstractSimulationHandlerPtr clone() const final;

private:
    details::VasculatureReportDetails _details;
    std::vector<doubles> _userData;
    bool _showVariations{false};

    common::SimulationReport _simulationReport;
};
using VasculatureHandlerPtr = std::shared_ptr<VasculatureHandler>;
} // namespace vasculature
} // namespace bioexplorer
