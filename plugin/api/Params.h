/* Copyright (c) 2020-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: cyrille.favreau@epfl.ch
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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

bool from_json(MeshDescriptor &param, const std::string &payload);
bool from_json(SugarsDescriptor &param, const std::string &payload);

// Other elements
bool from_json(AddGrid &param, const std::string &payload);

// Functions
bool from_json(AminoAcidSequenceAsStringDescriptor &param,
               const std::string &payload);
bool from_json(AminoAcidSequenceAsRangeDescriptor &param,
               const std::string &payload);
bool from_json(AminoAcidInformationDescriptor &param,
               const std::string &payload);

bool from_json(FileAccess &param, const std::string &payload);
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

} // namespace bioexplorer
