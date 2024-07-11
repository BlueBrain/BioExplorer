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

#include <plugin/api/SonataExplorerParams.h>
#include <plugin/neuroscience/common/Types.h>

#include <platform/core/common/Api.h>
#include <platform/core/common/Types.h>
#include <platform/core/common/simulation/AbstractAnimationHandler.h>
#include <platform/core/engineapi/Scene.h>

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
/**
 * @brief The VoltageSimulationHandler class handles simulation frames for the
 * current circuit. Frames are stored in a memory mapped file that is accessed
 * according to a specified timestamp. The VoltageSimulationHandler class is in
 * charge of keeping the handle to the memory mapped file.
 */
class VoltageSimulationHandler : public core::AbstractAnimationHandler
{
public:
    /**
     * @brief Default constructor
     * @param geometryParameters Geometry parameters
     * @param reportSource path to report source
     * @param gids GIDS to load
     */
    VoltageSimulationHandler(const std::string& reportPath, const brion::GIDSet& gids,
                             const bool synchronousMode = false);
    VoltageSimulationHandler(const VoltageSimulationHandler& rhs);
    ~VoltageSimulationHandler();

    void* getFrameData(const uint32_t frame) final;

    const std::string& getReportPath() const { return _reportPath; }
    common::CompartmentReportPtr getReport() const { return _compartmentReport; }
    bool isSynchronized() const { return _synchronousMode; }
    bool isReady() const final;

    core::AbstractSimulationHandlerPtr clone() const final;

private:
    void _triggerLoading(const uint32_t frame);
    bool _isFrameLoaded() const;
    bool _makeFrameReady(const uint32_t frame);
    bool _synchronousMode{false};

    std::string _reportPath;
    common::CompartmentReportPtr _compartmentReport;
    std::future<brion::Frame> _currentFrameFuture;
    uint64_t _startFrame{0};
    bool _ready{false};
};
using VoltageSimulationHandlerPtr = std::shared_ptr<VoltageSimulationHandler>;
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
