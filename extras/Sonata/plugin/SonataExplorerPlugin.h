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

#include <plugin/api/SonataExplorerParams.h>

#include <array>
#include <brayns/common/types.h>
#include <brayns/pluginapi/ExtensionPlugin.h>
#include <vector>

namespace sonataexplorer
{
using namespace brayns;
using namespace api;

/**
 * @brief The SonataExplorerPlugin class manages the loading and visualization
 * of the Blue Brain Project micro-circuits, and allows visualisation of voltage
 * simulations
 */
class SonataExplorerPlugin : public ExtensionPlugin
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
    Response _getVersion() const;
    void _markModified() { _dirty = true; };

    // Handlers
    Response _attachCellGrowthHandler(const AttachCellGrowthHandler& payload);
    Response _attachCircuitSimulationHandler(
        const AttachCircuitSimulationHandler& payload);
    Response _setConnectionsPerValue(const ConnectionsPerValue&);

    SynapseAttributes _synapseAttributes;

    // Rendering
    Response _setMaterial(const MaterialDescriptor&);
    Response _setMaterials(const MaterialsDescriptor&);
    Response _setMaterialRange(const MaterialRangeDescriptor&);
    Response _setMaterialExtraAttributes(const MaterialExtraAttributes&);
    MaterialIds _getMaterialIds(const ModelId& modelId);

    // Experimental
    Response _exportModelToFile(const ExportModelToFile&);
    Response _exportModelToMesh(const ExportModelToMesh&);

    // Add geometry
    void _createShapeMaterial(ModelPtr& model, const size_t id,
                              const Vector3d& color, const double& opacity);
    Response _addSphere(const AddSphere& payload);
    Response _addPill(const AddPill& payload);
    Response _addCylinder(const AddCylinder& payload);
    Response _addBox(const AddBox& payload);

    // Predefined models
    Response _addColumn(const AddColumn& payload);

    bool _dirty{false};
};
} // namespace sonataexplorer
