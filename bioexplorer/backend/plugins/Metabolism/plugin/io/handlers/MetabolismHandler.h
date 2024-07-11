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

#include <plugin/api/Params.h>
#include <plugin/common/Types.h>
#include <plugin/io/db/DBConnector.h>

#include <platform/core/common/Api.h>
#include <platform/core/common/simulation/AbstractAnimationHandler.h>

namespace bioexplorer
{
namespace metabolism
{
/**
 * @brief The MetabolismHandler class handles metabolite concentrations
 */
class MetabolismHandler : public core::AbstractAnimationHandler
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

    core::AbstractSimulationHandlerPtr clone() const final;

    void setMetaboliteIds(const int32_ts& metaboliteIds) { _metaboliteIds = metaboliteIds; }

private:
    DBConnectorPtr _connector{nullptr};

    int32_ts _metaboliteIds;
    Locations _locations;
    bool _relativeConcentration{false};
    uint32_t _referenceFrame{0};
};
} // namespace metabolism
} // namespace bioexplorer
