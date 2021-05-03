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

#include <plugin/common/Types.h>

using namespace bioexplorer;
using namespace details;

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
// Response
std::string to_json(const Response &param);

// Settings
bool from_json(GeneralSettingsDetails &param, const std::string &payload);

// Scene information
std::string to_json(const SceneInformationDetails &param);

// Biological elements
bool from_json(AssemblyDetails &param, const std::string &payload);
std::string to_json(const AssemblyDetails &payload);

bool from_json(AssemblyTransformationsDetails &param,
               const std::string &payload);

bool from_json(RNASequenceDetails &param, const std::string &payload);

bool from_json(ParametricMembraneDetails &param, const std::string &payload);
bool from_json(MeshBasedMembraneDetails &param, const std::string &payload);

bool from_json(ProteinDetails &param, const std::string &payload);
std::string to_json(const ProteinDetails &payload);

bool from_json(SugarsDetails &param, const std::string &payload);

// Other elements
bool from_json(AddGridDetails &param, const std::string &payload);
bool from_json(AddSphereDetails &param, const std::string &payload);

// Amino acids
bool from_json(AminoAcidSequenceAsStringDetails &param,
               const std::string &payload);
bool from_json(AminoAcidSequenceAsRangesDetails &param,
               const std::string &payload);
bool from_json(AminoAcidInformationDetails &param, const std::string &payload);
bool from_json(AminoAcidDetails &param, const std::string &payload);

// Files
bool from_json(FileAccessDetails &param, const std::string &payload);

// DB
bool from_json(DatabaseAccessDetails &param, const std::string &payload);

// Color schemes and materials
bool from_json(ColorSchemeDetails &param, const std::string &payload);
bool from_json(ModelIdDetails &modelId, const std::string &payload);
bool from_json(MaterialsDetails &materialsDetails, const std::string &payload);
std::string to_json(const MaterialIdsDetails &param);

// Fields
bool from_json(BuildFieldsDetails &param, const std::string &payload);
bool from_json(ModelIdFileAccessDetails &param, const std::string &payload);

// Point cloud
bool from_json(BuildPointCloudDetails &param, const std::string &payload);

// Models and instances
bool from_json(ModelsVisibilityDetails &param, const std::string &payload);
bool from_json(ProteinInstanceTransformationDetails &param,
               const std::string &payload);
#endif