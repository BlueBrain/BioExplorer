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

#include "BioExplorerParams.h"
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

bool from_json(AminoAcidSequenceAsRangeDescriptor &param,
               const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, range);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AminoAcidSequencesDescriptor &param, const std::string &payload)
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

bool from_json(RNASequenceDescriptor &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, contents);
        FROM_JSON(param, js, shape);
        FROM_JSON(param, js, assemblyRadius);
        FROM_JSON(param, js, radius);
        FROM_JSON(param, js, range);
        FROM_JSON(param, js, params);
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
        FROM_JSON(param, js, assemblyRadius);
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
        FROM_JSON(param, js, assemblyRadius);
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
        TO_JSON(payload, js, assemblyRadius);
        TO_JSON(payload, js, atomRadiusMultiplier);
        TO_JSON(payload, js, loadBonds);
        TO_JSON(payload, js, loadNonPolymerChemicals);
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
        FROM_JSON(param, js, addSticks);
        FROM_JSON(param, js, recenter);
        FROM_JSON(param, js, siteIndices);
        FROM_JSON(param, js, allowedOccurrences);
        FROM_JSON(param, js, orientation);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(MeshDescriptor &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, contents);
        FROM_JSON(param, js, shape);
        FROM_JSON(param, js, assemblyRadius);
        FROM_JSON(param, js, recenter);
        FROM_JSON(param, js, occurrences);
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

bool from_json(LoaderExportToFileDescriptor &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, filename);
    }
    catch (...)
    {
        return false;
    }
    return true;
}
} // namespace bioexplorer
