/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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

#include <science/api/Params.h>
#include <science/fields/PointFieldsHandler.h>
#include <science/fields/VectorFieldsHandler.h>
#include <science/vasculature/Vasculature.h>

#include <platform/core/pluginapi/ExtensionPlugin.h>

namespace bioexplorer
{
/**
 * @brief This class implements the BioExplorer plugin
 */
class BioExplorerPlugin : public core::ExtensionPlugin
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
    void _createRenderers();
#ifdef USE_OPTIX6
    void _createOptiXRenderers();
#endif

    // Info and settings
    details::Response _getVersion() const;
    details::SceneInformationDetails _getSceneInformation() const;
    details::Response _setGeneralSettings(const details::GeneralSettingsDetails &payload);
    details::Response _resetScene();
    details::Response _resetCamera();
    details::Response _setFocusOn(const details::FocusOnDetails &details);

    // IO
    details::Response _exportToFile(const details::FileAccessDetails &payload);
    details::Response _importFromFile(const details::FileAccessDetails &payload);
    details::Response _exportToXYZ(const details::FileAccessDetails &payload);

    // DB
    details::Response _exportBrickToDatabase(const details::DatabaseAccessDetails &payload);

    // Biological elements
    details::Response _addAssembly(const details::AssemblyDetails &payload);
    details::Response _removeAssembly(const details::AssemblyDetails &payload);
    details::Response _addMembrane(const details::MembraneDetails &payload) const;
    details::Response _addRNASequence(const details::RNASequenceDetails &payload) const;
    details::Response _addProtein(const details::ProteinDetails &payload) const;
    details::Response _addGlycan(const details::SugarDetails &payload) const;
    details::Response _addSugar(const details::SugarDetails &payload) const;
    details::Response _addEnzymeReaction(const details::EnzymeReactionDetails &payload) const;
    details::Response _setEnzymeReactionProgress(const details::EnzymeReactionProgressDetails &payload) const;

    // Other elements
    details::Response _addGrid(const details::AddGridDetails &payload);
    details::Response _addSpheres(const details::AddSpheresDetails &payload);
    details::Response _addCones(const details::AddConesDetails &payload);
    details::Response _addBoundingBox(const details::AddBoundingBoxDetails &payload);
    details::Response _addBox(const details::AddBoxDetails &payload);
    details::Response _addStreamlines(const details::AddStreamlinesDetails &payload);

    // Amino acids
    details::Response _setAminoAcidSequenceAsString(const details::AminoAcidSequenceAsStringDetails &payload) const;
    details::Response _setAminoAcidSequenceAsRanges(const details::AminoAcidSequenceAsRangesDetails &payload) const;
    details::Response _getAminoAcidInformation(const details::AminoAcidInformationDetails &payload) const;
    details::Response _setAminoAcid(const details::AminoAcidDetails &payload) const;

    // Portein instances
    details::Response _setProteinInstanceTransformation(
        const details::ProteinInstanceTransformationDetails &payload) const;
    details::Response _getProteinInstanceTransformation(
        const details::ProteinInstanceTransformationDetails &payload) const;

    // Models
    details::NameDetails _getModelName(const details::ModelIdDetails &payload) const;
    details::ModelTransformationDetails _getModelTransformation(const details::ModelIdDetails &payload) const;
    details::ModelBoundsDetails _getModelBounds(const details::ModelIdDetails &payload) const;
    details::IdsDetails _getModelIds() const;
    details::IdsDetails _getModelInstances(const details::ModelIdDetails &payload) const;
    details::Response _addModelInstance(const details::AddModelInstanceDetails &payload) const;
    details::Response _setModelInstances(const details::SetModelInstancesDetails &payload) const;

    // Colors and materials
    details::Response _setProteinColorScheme(const details::ProteinColorSchemeDetails &payload) const;
    details::Response _setMaterials(const details::MaterialsDetails &payload);
    details::IdsDetails _getMaterialIds(const details::ModelIdDetails &payload);

    // Point clouds
    details::Response _buildPointCloud(const details::BuildPointCloudDetails &payload);

    // Fields
    details::Response _buildFields(const details::BuildFieldsDetails &payload);

    // Models
    details::Response _setModelLoadingTransactionAction(const details::ModelLoadingTransactionDetails &payload);

    // Out-Of-Core
    details::Response _getOOCConfiguration() const;
    details::Response _getOOCProgress() const;
    details::Response _getOOCAverageLoadingTime() const;
    io::OOCManagerPtr _oocManager{nullptr};

    // Inspection
    details::ProteinInspectionDetails _inspectProtein(const details::InspectionDetails &details) const;

    // Atlas
    details::Response _addAtlas(const details::AtlasDetails &payload);

    // Vasculature
    details::Response _addVasculature(const details::VasculatureDetails &payload);
    details::Response _getVasculatureInfo(const details::NameDetails &payload) const;
    details::Response _setVasculatureReport(const details::VasculatureReportDetails &payload);
    details::Response _setVasculatureRadiusReport(const details::VasculatureRadiusReportDetails &payload);

    // Astrocytes
    details::Response _addAstrocytes(const details::AstrocytesDetails &payload);

    // Neurons
    details::Response _addNeurons(const details::NeuronsDetails &payload);
    details::NeuronPointsDetails _getNeuronSectionPoints(const details::NeuronIdSectionIdDetails &payload);
    details::NeuronPointsDetails _getNeuronVaricosities(const details::NeuronIdDetails &payload);

    // Connectomics
    details::Response _addSynaptome(const details::SynaptomeDetails &payload);
    details::Response _addWhiteMatter(const details::WhiteMatterDetails &payload);
    details::Response _addSynapses(const details::SynapsesDetails &payload);
    details::Response _addSynapseEfficacy(const details::SynapseEfficacyDetails &payload);
    details::Response _setSpikeReportVisualizationSettings(
        const details::SpikeReportVisualizationSettingsDetails &payload);

    // Utilities
    details::LookAtResponseDetails _lookAt(const details::LookAtDetails &payload);
    details::Response _addSdfDemo();

    // SDF Geometries
    details::Response _addTorus(const details::SDFTorusDetails &payload);
    details::Response _addVesica(const details::SDFVesicaDetails &payload);

    // Attributes
    common::AssemblyMap _assemblies;

    // Command line arguments
    std::map<std::string, std::string> _commandLineArguments;
};
} // namespace bioexplorer
