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

#include <plugin/api/SonataExplorerParams.h>
#include <plugin/neuroscience/common/Types.h>

#include <platform/core/common/Api.h>
#include <platform/core/common/Types.h>
#include <platform/core/common/simulation/AbstractSimulationHandler.h>
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
class VoltageSimulationHandler : public core::AbstractSimulationHandler
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
