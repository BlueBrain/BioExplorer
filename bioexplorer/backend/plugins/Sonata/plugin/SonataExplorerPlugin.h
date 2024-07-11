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
    SonataExplorerPlugin(int argc, char** argv);

    void init() final;

    /**
     * @brief preRender Updates the scene according to latest data load
     */
    void preRender() final;

private:
    // Plug-in
    api::Response _getVersion() const;
    void _parseCommandLineArguments(int argc, char** argv);
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

    // Import to DB
    api::Response _importCircuitMorphologies(const api::ImportCircuitMorphologies& payload);

    // Command line arguments
    std::map<std::string, std::string> _commandLineArguments;

    bool _dirty{false};
};
} // namespace sonataexplorer
