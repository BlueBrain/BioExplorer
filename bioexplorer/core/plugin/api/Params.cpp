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

#include "Params.h"
#include "json.hpp"

namespace bioexplorer
{
#ifndef BRAYNS_DEBUG_JSON_ENABLED
#define FROM_JSON(PARAM, JSON, NAME) \
    PARAM.NAME = JSON[#NAME].get<decltype(PARAM.NAME)>()
#else
#define FROM_JSON(PARAM, JSON, NAME)                                          \
    try                                                                       \
    {                                                                         \
        PARAM.NAME = JSON[#NAME].get<decltype(PARAM.NAME)>();                 \
    }                                                                         \
    catch (...)                                                               \
    {                                                                         \
        PLUGIN_ERROR << "JSON parsing error for attribute '" << #NAME << "'!" \
                     << std::endl;                                            \
        throw;                                                                \
    }
#endif
#define TO_JSON(PARAM, JSON, NAME) JSON[#NAME] = PARAM.NAME

std::string to_json(const Response &param)
{
    try
    {
        nlohmann::json js;

        TO_JSON(param, js, status);
        TO_JSON(param, js, contents);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

bool from_json(AssemblyDescriptor &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, position);
        FROM_JSON(param, js, clippingPlanes);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

std::string to_json(const AssemblyDescriptor &payload)
{
    try
    {
        nlohmann::json js;

        TO_JSON(payload, js, name);
        TO_JSON(payload, js, position);
        TO_JSON(payload, js, orientation);
        TO_JSON(payload, js, clippingPlanes);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

bool from_json(AssemblyTransformationsDescriptor &param,
               const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, transformations);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ColorSchemeDescriptor &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, colorScheme);
        FROM_JSON(param, js, palette);
        FROM_JSON(param, js, chainIds);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AminoAcidSequenceAsStringDescriptor &param,
               const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, sequence);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AminoAcidSequenceAsRangesDescriptor &param,
               const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, ranges);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AminoAcidInformationDescriptor &param,
               const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(SetAminoAcid &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, index);
        FROM_JSON(param, js, aminoAcidShortName);
        FROM_JSON(param, js, chainIds);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(RNASequenceDescriptor &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, contents);
        FROM_JSON(param, js, shape);
        FROM_JSON(param, js, assemblyParams);
        FROM_JSON(param, js, range);
        FROM_JSON(param, js, params);
        FROM_JSON(param, js, position);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(MembraneDescriptor &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, content1);
        FROM_JSON(param, js, content2);
        FROM_JSON(param, js, content3);
        FROM_JSON(param, js, content4);
        FROM_JSON(param, js, shape);
        FROM_JSON(param, js, assemblyParams);
        FROM_JSON(param, js, atomRadiusMultiplier);
        FROM_JSON(param, js, loadBonds);
        FROM_JSON(param, js, loadNonPolymerChemicals);
        FROM_JSON(param, js, representation);
        FROM_JSON(param, js, chainIds);
        FROM_JSON(param, js, recenter);
        FROM_JSON(param, js, occurrences);
        FROM_JSON(param, js, randomSeed);
        FROM_JSON(param, js, locationCutoffAngle);
        FROM_JSON(param, js, positionRandomizationType);
        FROM_JSON(param, js, orientation);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ProteinDescriptor &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, contents);
        FROM_JSON(param, js, shape);
        FROM_JSON(param, js, assemblyParams);
        FROM_JSON(param, js, atomRadiusMultiplier);
        FROM_JSON(param, js, loadBonds);
        FROM_JSON(param, js, loadNonPolymerChemicals);
        FROM_JSON(param, js, representation);
        FROM_JSON(param, js, chainIds);
        FROM_JSON(param, js, recenter);
        FROM_JSON(param, js, occurrences);
        FROM_JSON(param, js, allowedOccurrences);
        FROM_JSON(param, js, randomSeed);
        FROM_JSON(param, js, locationCutoffAngle);
        FROM_JSON(param, js, positionRandomizationType);
        FROM_JSON(param, js, position);
        FROM_JSON(param, js, orientation);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

std::string to_json(const ProteinDescriptor &payload)
{
    try
    {
        nlohmann::json js;

        TO_JSON(payload, js, assemblyName);
        TO_JSON(payload, js, name);
        TO_JSON(payload, js, contents);
        TO_JSON(payload, js, shape);
        TO_JSON(payload, js, assemblyParams);
        TO_JSON(payload, js, atomRadiusMultiplier);
        TO_JSON(payload, js, loadBonds);
        TO_JSON(payload, js, loadNonPolymerChemicals);
        TO_JSON(payload, js, loadHydrogen);
        TO_JSON(payload, js, representation);
        TO_JSON(payload, js, chainIds);
        TO_JSON(payload, js, recenter);
        TO_JSON(payload, js, occurrences);
        TO_JSON(payload, js, allowedOccurrences);
        TO_JSON(payload, js, randomSeed);
        TO_JSON(payload, js, locationCutoffAngle);
        TO_JSON(payload, js, positionRandomizationType);
        TO_JSON(payload, js, position);
        TO_JSON(payload, js, orientation);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

bool from_json(SugarsDescriptor &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, contents);
        FROM_JSON(param, js, proteinName);
        FROM_JSON(param, js, atomRadiusMultiplier);
        FROM_JSON(param, js, loadBonds);
        FROM_JSON(param, js, representation);
        FROM_JSON(param, js, recenter);
        FROM_JSON(param, js, chainIds);
        FROM_JSON(param, js, siteIndices);
        FROM_JSON(param, js, orientation);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(MeshBasedMembraneDescriptor &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, meshContents);
        FROM_JSON(param, js, proteinContents);
        FROM_JSON(param, js, recenter);
        FROM_JSON(param, js, density);
        FROM_JSON(param, js, surfaceFixedOffset);
        FROM_JSON(param, js, surfaceVariableOffset);
        FROM_JSON(param, js, atomRadiusMultiplier);
        FROM_JSON(param, js, representation);
        FROM_JSON(param, js, randomSeed);
        FROM_JSON(param, js, position);
        FROM_JSON(param, js, orientation);
        FROM_JSON(param, js, scale);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AddGrid &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, minValue);
        FROM_JSON(param, js, maxValue);
        FROM_JSON(param, js, steps);
        FROM_JSON(param, js, radius);
        FROM_JSON(param, js, planeOpacity);
        FROM_JSON(param, js, showAxis);
        FROM_JSON(param, js, useColors);
        FROM_JSON(param, js, position);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ModelId &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(MaterialsDescriptor &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelIds);
        FROM_JSON(param, js, materialIds);
        FROM_JSON(param, js, diffuseColors);
        FROM_JSON(param, js, specularColors);
        FROM_JSON(param, js, specularExponents);
        FROM_JSON(param, js, reflectionIndices);
        FROM_JSON(param, js, opacities);
        FROM_JSON(param, js, refractionIndices);
        FROM_JSON(param, js, emissions);
        FROM_JSON(param, js, glossinesses);
        FROM_JSON(param, js, shadingModes);
        FROM_JSON(param, js, userParameters);
        FROM_JSON(param, js, chameleonModes);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

std::string to_json(const MaterialIds &param)
{
    try
    {
        nlohmann::json js;
        TO_JSON(param, js, ids);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

// Fields
bool from_json(BuildFields &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, voxelSize);
        FROM_JSON(param, js, density);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(FileAccess &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, filename);
        FROM_JSON(param, js, fileFormat);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ModelIdFileAccess &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
        FROM_JSON(param, js, filename);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(BuildPointCloud &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, radius);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

} // namespace bioexplorer
