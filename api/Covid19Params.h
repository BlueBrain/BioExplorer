/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#ifndef COVID19_PARAMS_H
#define COVID19_PARAMS_H

#include <common/types.h>

#include <set>

/** Define the color scheme to be applied to the geometry */

struct Response
{
    bool status{true};
    std::string contents;
};
std::string to_json(const Response &param);

struct StructureDescriptor
{
    std::string name;
    std::string path;
    size_t occurrences;
    float assemblyRadius;
    float atomRadiusMultiplier;
    size_t randomSeed;
    bool halfStructure;
    std::vector<float> upVector;
};
bool from_json(StructureDescriptor &param, const std::string &payload);

struct ColorSchemeDescriptor
{
    std::string path;
    ColorScheme colorScheme;
    std::vector<float> palette;
};
bool from_json(ColorSchemeDescriptor &param, const std::string &payload);

struct AminoAcidSequenceDescriptor
{
    std::string path;
    std::string aminoAcidSequence;
};
bool from_json(AminoAcidSequenceDescriptor &param, const std::string &payload);

struct AminoAcidSequencesDescriptor
{
    std::string path;
};
bool from_json(AminoAcidSequencesDescriptor &param, const std::string &payload);

struct RNADescriptor
{
    std::string name;
    std::string path;
    RNAShape shape;
    float assemblyRadius;
    float radius;
    std::vector<float> range;
    std::vector<float> params;
};
bool from_json(RNADescriptor &param, const std::string &payload);

#endif // COVID19_PARAMS_H
