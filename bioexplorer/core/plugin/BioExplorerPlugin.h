/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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
#include <plugin/vasculature/Vasculature.h>

#include <brayns/pluginapi/ExtensionPlugin.h>

namespace bioexplorer
{
using namespace fields;
using namespace molecularsystems;
using namespace vasculature;
using namespace details;
using namespace io;

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
    BioExplorerPlugin(int argc, char **argv);

    /**
     * @brief Plugin initialization, registration of end-points, renderers,
     * cameras, etc.
     *
     */
    void init() final;

    void preRender() final;

private:
    // Command line arguments
    void _parseCommandLineArguments(int argc, char **argv);

    // Info and settings
    Response _getVersion() const;
    SceneInformationDetails _getSceneInformation() const;
    Response _setGeneralSettings(const GeneralSettingsDetails &payload);
    Response _resetScene();
    Response _resetCamera();
    Response _setFocusOn(const FocusOnDetails &details);

    // IO
    Response _exportToFile(const FileAccessDetails &payload);
    Response _importFromFile(const FileAccessDetails &payload);
    Response _exportToXYZ(const FileAccessDetails &payload);

    // DB
    Response _exportBrickToDatabase(const DatabaseAccessDetails &payload);

    // Biological elements
    Response _addAssembly(const AssemblyDetails &payload);
    Response _removeAssembly(const AssemblyDetails &payload);
    Response _addMembrane(const MembraneDetails &payload) const;
    Response _addRNASequence(const RNASequenceDetails &payload) const;
    Response _addProtein(const ProteinDetails &payload) const;
    Response _addGlycans(const SugarsDetails &payload) const;
    Response _addSugars(const SugarsDetails &payload) const;

    // Other elements
    Response _addGrid(const AddGridDetails &payload);
    Response _addSphere(const AddSphereDetails &payload);
    Response _addBoundingBox(const AddBoundingBoxDetails &payload);

    // Amino acids
    Response _setAminoAcidSequenceAsString(
        const AminoAcidSequenceAsStringDetails &payload) const;
    Response _setAminoAcidSequenceAsRanges(
        const AminoAcidSequenceAsRangesDetails &payload) const;
    Response _getAminoAcidInformation(
        const AminoAcidInformationDetails &payload) const;
    Response _setAminoAcid(const AminoAcidDetails &payload) const;

    // Portein instances
    Response _setProteinInstanceTransformation(
        const ProteinInstanceTransformationDetails &payload) const;
    Response _getProteinInstanceTransformation(
        const ProteinInstanceTransformationDetails &payload) const;

    // Models
    NameDetails _getModelName(const ModelIdDetails &payload) const;
    IdsDetails _getModelIds() const;
    IdsDetails _getModelInstances(const ModelIdDetails &payload) const;

    // Colors and materials
    Response _setProteinColorScheme(
        const ProteinColorSchemeDetails &payload) const;
    Response _setMaterials(const MaterialsDetails &payload);
    IdsDetails _getMaterialIds(const ModelIdDetails &payload);

    // Point clouds
    Response _buildPointCloud(const BuildPointCloudDetails &payload);

    // Fields
    size_t _attachFieldsHandler(FieldsHandlerPtr handler);
    Response _buildFields(const BuildFieldsDetails &payload);
    Response _exportFieldsToFile(const ModelIdFileAccessDetails &payload);
    Response _importFieldsFromFile(const FileAccessDetails &payload);

    // Models
    Response _setModelsVisibility(const ModelsVisibilityDetails &payload);

    // Out-Of-Core
    Response _getOOCConfiguration() const;
    Response _getOOCProgress() const;
    Response _getOOCAverageLoadingTime() const;
    OOCManagerPtr _oocManager{nullptr};

    // Inspection
    ProteinInspectionDetails _inspectProtein(
        const InspectionDetails &details) const;

    // Attributes
    AssemblyMap _assemblies;

    // Command line arguments
    std::map<std::string, std::string> _commandLineArguments;

#ifdef USE_VASCULATURE
    // Vasculature
    Response _addVasculature(const VasculatureDetails &payload);
    Response _getVasculatureInfo(const NameDetails &payload) const;
    Response _setVasculatureColorScheme(
        const VasculatureColorSchemeDetails &payload);
    Response _setVasculatureReport(const VasculatureReportDetails &payload);
    Response _setVasculatureRadiusReport(
        const VasculatureRadiusReportDetails &payload);
#endif
};
} // namespace bioexplorer
