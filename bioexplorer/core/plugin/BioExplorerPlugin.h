/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#include <plugin/api/Params.h>
#include <plugin/fields/FieldsHandler.h>

#include <brayns/pluginapi/ExtensionPlugin.h>

namespace bioexplorer
{
/**
 * @brief This class implements the BioExplorer plugin
 */
class BioExplorerPlugin : public ExtensionPlugin
{
public:
    /**
     * @brief Construct a new Bio Explorer Plugin object
     *
     */
    BioExplorerPlugin();

    /**
     * @brief
     *
     */
    void init() final;

    /**
     * @brief
     *
     */
    void preRender() final;

private:
    Response _version() const;

    // IO
    Response _exportToCache(const FileAccess &payload);
    Response _exportToXYZ(const FileAccess &payload);

    // Biological elements
    Response _addAssembly(const AssemblyDescriptor &payload);
    Response _removeAssembly(const AssemblyDescriptor &payload);
    Response _addMembrane(const MembraneDescriptor &payload) const;
    Response _addRNASequence(const RNASequenceDescriptor &payload) const;
    Response _addProtein(const ProteinDescriptor &payload) const;
    Response _addMesh(const MeshDescriptor &payload) const;
    Response _addGlycans(const SugarsDescriptor &payload) const;
    Response _addSugars(const SugarsDescriptor &payload) const;

    // Other elements
    Response _addGrid(const AddGrid &payload);

    // Amino acids
    Response _setAminoAcidSequenceAsString(
        const AminoAcidSequenceAsStringDescriptor &payload) const;
    Response _setAminoAcidSequenceAsRanges(
        const AminoAcidSequenceAsRangesDescriptor &payload) const;
    Response _getAminoAcidInformation(
        const AminoAcidInformationDescriptor &payload) const;
    Response _setAminoAcid(const SetAminoAcid &payload) const;

    // Colors and materials
    Response _setColorScheme(const ColorSchemeDescriptor &payload) const;
    Response _setMaterials(const MaterialsDescriptor &payload);
    MaterialIds _getMaterialIds(const ModelId &modelId);

    // Point clouds
    Response _buildPointCloud(const BuildPointCloud &payload);

    AssemblyMap _assemblies;
    bool _dirty{false};

    // Fields
    void _attachFieldsHandler(FieldsHandlerPtr handler);
    Response _buildFields(const BuildFields &payload);
    Response _exportFieldsToFile(const ModelIdFileAccess &payload);
    Response _importFieldsFromFile(const FileAccess &payload);
};
} // namespace bioexplorer
