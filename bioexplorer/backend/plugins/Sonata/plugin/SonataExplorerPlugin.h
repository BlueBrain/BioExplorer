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

#include <array>
#include <platform/core/common/Types.h>
#include <platform/core/pluginapi/ExtensionPlugin.h>
#include <vector>

namespace sonataexplorer
{
/**
 * @brief The SonataExplorerPlugin class manages the loading and visualization
 * of the Blue Brain Project micro-circuits, and allows visualisation of voltage
 * simulations
 */
class SonataExplorerPlugin : public core::ExtensionPlugin
{
public:
    SonataExplorerPlugin();

    void init() final;

    /**
     * @brief preRender Updates the scene according to latest data load
     */
    void preRender() final;

private:
    // Plug-in
    api::Response _getVersion() const;
    void _markModified() { _dirty = true; };

    // Handlers
    api::Response _attachCellGrowthHandler(const api::AttachCellGrowthHandler& payload);
    api::Response _attachCircuitSimulationHandler(const api::AttachCircuitSimulationHandler& payload);
    api::Response _setConnectionsPerValue(const api::ConnectionsPerValue&);
    api::Response _setSpikeReportVisualizationSettings(const api::SpikeReportVisualizationSettings& payload);

    api::SynapseAttributes _synapseAttributes;

    // Experimental
    api::Response _exportModelToFile(const api::ExportModelToFile&);
    api::Response _exportModelToMesh(const api::ExportModelToMesh&);

    // Add geometry
    void _createShapeMaterial(core::ModelPtr& model, const size_t id, const core::Vector3d& color,
                              const double& opacity);
    api::Response _addSphere(const api::AddSphere& payload);
    api::Response _addPill(const api::AddPill& payload);
    api::Response _addCylinder(const api::AddCylinder& payload);
    api::Response _addBox(const api::AddBox& payload);

    // Predefined models
    api::Response _addColumn(const api::AddColumn& payload);

    // MEG
    api::Response _loadMEG(const api::LoadMEGSettings& payload);

    // Morphology cache
    api::Response _enabledMorphologyCache(const api::EnableMorphologyCache& payload);

    bool _dirty{false};
};
} // namespace sonataexplorer
