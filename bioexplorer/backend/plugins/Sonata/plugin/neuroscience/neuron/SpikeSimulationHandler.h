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

#include <brain/brain.h>
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
typedef std::shared_ptr<brain::SpikeReportReader> SpikeReportReaderPtr;

class SpikeSimulationHandler : public core::AbstractAnimationHandler
{
public:
    SpikeSimulationHandler(const std::string& reportPath, const brain::GIDSet& gids);
    SpikeSimulationHandler(const SpikeSimulationHandler& rhs);

    void* getFrameData(const uint32_t frame) final;

    const std::string& getReportPath() const { return _reportPath; }
    SpikeReportReaderPtr getReport() const { return _spikeReport; }
    const brain::GIDSet& getGIDs() const { return _gids; }

    core::AbstractSimulationHandlerPtr clone() const final;

    void setVisualizationSettings(const double restVoltage, const double spikingVoltage, const double timeInterval,
                                  const double decaySpeed);

private:
    void _logVisualizationSettings();

    std::string _reportPath;
    brain::GIDSet _gids;
    SpikeReportReaderPtr _spikeReport;

    double _restVoltage{-65.0};
    double _spikingVoltage{-10.0};
    double _timeInterval{0.01};
    double _decaySpeed{1.0};

    std::map<uint64_t, uint64_t> _gidMap;
};
using SpikeSimulationHandlerPtr = std::shared_ptr<SpikeSimulationHandler>;
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
