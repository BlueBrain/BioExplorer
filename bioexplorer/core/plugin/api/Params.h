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

namespace bioexplorer
{
struct Response
{
    bool status{true};
    std::string contents;
};
std::string to_json(const Response &param);

// Biological elements
bool from_json(AssemblyDescriptor &param, const std::string &payload);
std::string to_json(const AssemblyDescriptor &payload);

bool from_json(AssemblyTransformationsDescriptor &param,
               const std::string &payload);

bool from_json(RNASequenceDescriptor &param, const std::string &payload);

bool from_json(MembraneDescriptor &param, const std::string &payload);

bool from_json(ProteinDescriptor &param, const std::string &payload);
std::string to_json(const ProteinDescriptor &payload);

bool from_json(MeshBasedMembraneDescriptor &param, const std::string &payload);
bool from_json(SugarsDescriptor &param, const std::string &payload);

// Other elements
bool from_json(AddGrid &param, const std::string &payload);

// Amino acids
bool from_json(AminoAcidSequenceAsStringDescriptor &param,
               const std::string &payload);
bool from_json(AminoAcidSequenceAsRangesDescriptor &param,
               const std::string &payload);
bool from_json(AminoAcidInformationDescriptor &param,
               const std::string &payload);
bool from_json(SetAminoAcid &param, const std::string &payload);

// Files
bool from_json(FileAccess &param, const std::string &payload);

// Color schemes and materials
bool from_json(ColorSchemeDescriptor &param, const std::string &payload);
bool from_json(ModelId &modelId, const std::string &payload);
bool from_json(MaterialsDescriptor &materialsDescriptor,
               const std::string &payload);
std::string to_json(const MaterialIds &param);

// Fields
bool from_json(BuildFields &param, const std::string &payload);
bool from_json(ModelIdFileAccess &param, const std::string &payload);

// Point cloud
bool from_json(BuildPointCloud &param, const std::string &payload);

} // namespace bioexplorer
