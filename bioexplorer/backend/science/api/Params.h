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
bool from_json(bioexplorer::details::SynaptomeDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::WhiteMatterDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::SynapsesDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::SynapseEfficacyDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::SpikeReportVisualizationSettingsDetails &param, const std::string &payload);

// Extra geometry
bool from_json(bioexplorer::details::SDFTorusDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::SDFVesicaDetails &param, const std::string &payload);
bool from_json(bioexplorer::details::SDFEllipsoidDetails &param, const std::string &payload);

// Utilities
bool from_json(bioexplorer::details::LookAtDetails &param, const std::string &payload);
std::string to_json(const bioexplorer::details::LookAtResponseDetails &param);
#endif
