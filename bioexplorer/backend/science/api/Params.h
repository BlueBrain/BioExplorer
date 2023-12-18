/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

#include <science/common/Types.h>

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
// Response
std::string to_json(const bioexplorer::details::Response &param);

// Settings
bool from_json(bioexplorer::details::GeneralSettingsDetails &param, const std::string &payload);

// Scene information
std::string to_json(const bioexplorer::details::SceneInformationDetails &param);

// Camera
bool from_json(bioexplorer::details::FocusOnDetails &param, const std::string &payload);

// Biological elements
bool from_json(bioexplorer::details::AssemblyDetails &param, const std::string &payload);
std::string to_json(const bioexplorer::details::AssemblyDetails &payload);

bool from_json(bioexplorer::details::AssemblyTransformationsDetails &param, const std::string &payload);

bool from_json(bioexplorer::details::RNASequenceDetails &param, const std::string &payload);

bool from_json(bioexplorer::details::MembraneDetails &param, const std::string &payload);

bool from_json(bioexplorer::details::ProteinDetails &param, const std::string &payload);
std::string to_json(const bioexplorer::details::ProteinDetails &payload);

bool from_json(bioexplorer::details::SugarDetails &param, const std::string &payload);

// Enzyme reactions
bool from_json(bioexplorer::details::EnzymeReactionDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::EnzymeReactionProgressDetails &param, const std::string &payload);

// Other elements
bool from_json(bioexplorer::details::AddGridDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::AddSpheresDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::AddConesDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::AddBoundingBoxDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::AddBoxDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::AddStreamlinesDetails &param, const std::string &payload);

// Amino acids
bool from_json(bioexplorer::details::AminoAcidSequenceAsStringDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::AminoAcidSequenceAsRangesDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::AminoAcidInformationDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::AminoAcidDetails &param, const std::string &payload);

// Files
bool from_json(bioexplorer::details::FileAccessDetails &param, const std::string &payload);

// DB
bool from_json(bioexplorer::details::DatabaseAccessDetails &param, const std::string &payload);

// Models, Color schemes and materials
bool from_json(bioexplorer::details::ProteinColorSchemeDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::ModelIdDetails &modelId, const std::string &payload);
bool from_json(bioexplorer::details::AddModelInstanceDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::SetModelInstancesDetails &param, const std::string &payload);
std::string to_json(const bioexplorer::details::ModelBoundsDetails &param);
std::string to_json(const bioexplorer::details::ModelTransformationDetails &param);
bool from_json(bioexplorer::details::MaterialsDetails &materialsDetails, const std::string &payload);
std::string to_json(const bioexplorer::details::IdsDetails &param);
bool from_json(bioexplorer::details::NameDetails &param, const std::string &payload);
std::string to_json(const bioexplorer::details::NameDetails &param);

// Fields
bool from_json(bioexplorer::details::BuildFieldsDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::ModelIdFileAccessDetails &param, const std::string &payload);

// Point cloud
bool from_json(bioexplorer::details::BuildPointCloudDetails &param, const std::string &payload);

// Models and instances
bool from_json(bioexplorer::details::ModelLoadingTransactionDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::ProteinInstanceTransformationDetails &param, const std::string &payload);

// Protein inspection
bool from_json(bioexplorer::details::InspectionDetails &param, const std::string &payload);
std::string to_json(const bioexplorer::details::ProteinInspectionDetails &param);

// Vasculature
bool from_json(bioexplorer::details::VasculatureDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::VasculatureReportDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::VasculatureRadiusReportDetails &param, const std::string &payload);

bool from_json(bioexplorer::details::AtlasDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::AstrocytesDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::NeuronsDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::NeuronIdSectionIdDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::NeuronIdDetails &param, const std::string &payload);
std::string to_json(const bioexplorer::details::NeuronPointsDetails &param);

// Connectomics
bool from_json(bioexplorer::details::GraphDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::WhiteMatterDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::SynapsesDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::SynapseEfficacyDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::SpikeReportVisualizationSettingsDetails &param, const std::string &payload);

// Extra geometry
bool from_json(bioexplorer::details::SDFTorusDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::SDFVesicaDetails &param, const std::string &payload);

// Utilities
bool from_json(bioexplorer::details::LookAtDetails &param, const std::string &payload);
std::string to_json(const bioexplorer::details::LookAtResponseDetails &param);
#endif
