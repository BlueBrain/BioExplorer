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

#include <brain/brain.h>
#include <brayns/api.h>
#include <brayns/common/simulation/AbstractSimulationHandler.h>
#include <brayns/common/types.h>
#include <brayns/engineapi/Scene.h>

namespace sonataexplorer
{
namespace neuroscience
{
namespace neuron
{
using namespace brayns;

typedef std::shared_ptr<brain::SpikeReportReader> SpikeReportReaderPtr;

class SpikeSimulationHandler : public AbstractSimulationHandler
{
public:
    SpikeSimulationHandler(const std::string& reportPath,
                           const brain::GIDSet& gids);
    SpikeSimulationHandler(const SpikeSimulationHandler& rhs);

    void* getFrameData(const uint32_t frame) final;

    const std::string& getReportPath() const { return _reportPath; }
    SpikeReportReaderPtr getReport() const { return _spikeReport; }
    const brain::GIDSet& getGIDs() const { return _gids; }

    AbstractSimulationHandlerPtr clone() const final;

    void setVisualizationSettings(const double restVoltage,
                                  const double spikingVoltage,
                                  const double timeInterval,
                                  const double decaySpeed);

private:
    void _logVisualizationSettings();

    std::string _reportPath;
    brain::GIDSet _gids;
    SpikeReportReaderPtr _spikeReport;

    double _restVoltage{-80.0};
    double _spikingVoltage{-10.0};
    double _timeInterval{0.01};
    double _decaySpeed{1.0};

    std::map<uint64_t, uint64_t> _gidMap;
};
using SpikeSimulationHandlerPtr = std::shared_ptr<SpikeSimulationHandler>;
} // namespace neuron
} // namespace neuroscience
} // namespace sonataexplorer
