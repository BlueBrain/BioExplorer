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

#include "Params.h"
#include "json.hpp"

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)

#ifndef BRAYNS_DEBUG_JSON_ENABLED
#define FROM_JSON(PARAM, JSON, NAME) \
    PARAM.NAME = JSON[#NAME].get<decltype(PARAM.NAME)>()
#else
#define FROM_JSON(PARAM, JSON, NAME)                                         \
    try                                                                      \
    {                                                                        \
        PARAM.NAME = JSON[#NAME].get<decltype(PARAM.NAME)>();                \
    }                                                                        \
    catch (...)                                                              \
    {                                                                        \
        PLUGIN_ERROR("JSON parsing error for attribute '" << #NAME << "'!"); \
        throw;                                                               \
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

std::string to_json(const SceneInformationDetails &param)
{
    try
    {
        nlohmann::json js;
        TO_JSON(param, js, nbModels);
        TO_JSON(param, js, nbMaterials);
        TO_JSON(param, js, nbSpheres);
        TO_JSON(param, js, nbCylinders);
        TO_JSON(param, js, nbCones);
        TO_JSON(param, js, nbVertices);
        TO_JSON(param, js, nbIndices);
        TO_JSON(param, js, nbNormals);
        TO_JSON(param, js, nbColors);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

bool from_json(GeneralSettingsDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelVisibilityOnCreation);
        FROM_JSON(param, js, meshFolder);
        FROM_JSON(param, js, loggingLevel);
        FROM_JSON(param, js, v1Compatibility);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(FocusOnDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
        FROM_JSON(param, js, instanceId);
        FROM_JSON(param, js, direction);
        FROM_JSON(param, js, distance);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AssemblyDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, shape);
        FROM_JSON(param, js, shapeParams);
        FROM_JSON(param, js, shapeMeshContents);
        FROM_JSON(param, js, position);
        FROM_JSON(param, js, rotation);
        FROM_JSON(param, js, clippingPlanes);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

std::string to_json(const AssemblyDetails &payload)
{
    try
    {
        nlohmann::json js;

        TO_JSON(payload, js, name);
        TO_JSON(payload, js, position);
        TO_JSON(payload, js, rotation);
        TO_JSON(payload, js, clippingPlanes);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

bool from_json(AssemblyTransformationsDetails &param,
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

bool from_json(ProteinColorSchemeDetails &param, const std::string &payload)
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

bool from_json(AminoAcidSequenceAsStringDetails &param,
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

bool from_json(AminoAcidSequenceAsRangesDetails &param,
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

bool from_json(AminoAcidInformationDetails &param, const std::string &payload)
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

bool from_json(AminoAcidDetails &param, const std::string &payload)
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

bool from_json(RNASequenceDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, pdbId);
        FROM_JSON(param, js, contents);
        FROM_JSON(param, js, proteinContents);
        FROM_JSON(param, js, shape);
        FROM_JSON(param, js, shapeParams);
        FROM_JSON(param, js, valuesRange);
        FROM_JSON(param, js, curveParams);
        FROM_JSON(param, js, atomRadiusMultiplier);
        FROM_JSON(param, js, representation);
        FROM_JSON(param, js, position);
        FROM_JSON(param, js, rotation);
        FROM_JSON(param, js, animationParams);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(MembraneDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, lipidPDBIds);
        FROM_JSON(param, js, lipidContents);
        FROM_JSON(param, js, lipidRotation);
        FROM_JSON(param, js, lipidDensity);
        FROM_JSON(param, js, atomRadiusMultiplier);
        FROM_JSON(param, js, loadBonds);
        FROM_JSON(param, js, loadNonPolymerChemicals);
        FROM_JSON(param, js, representation);
        FROM_JSON(param, js, chainIds);
        FROM_JSON(param, js, recenter);
        FROM_JSON(param, js, animationParams);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ProteinDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, pdbId);
        FROM_JSON(param, js, contents);
        FROM_JSON(param, js, atomRadiusMultiplier);
        FROM_JSON(param, js, loadBonds);
        FROM_JSON(param, js, loadNonPolymerChemicals);
        FROM_JSON(param, js, loadHydrogen);
        FROM_JSON(param, js, representation);
        FROM_JSON(param, js, chainIds);
        FROM_JSON(param, js, recenter);
        FROM_JSON(param, js, transmembraneParams);
        FROM_JSON(param, js, occurrences);
        FROM_JSON(param, js, allowedOccurrences);
        FROM_JSON(param, js, animationParams);
        FROM_JSON(param, js, position);
        FROM_JSON(param, js, rotation);
        FROM_JSON(param, js, constraints);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

std::string to_json(const ProteinDetails &payload)
{
    try
    {
        nlohmann::json js;

        TO_JSON(payload, js, assemblyName);
        TO_JSON(payload, js, name);
        TO_JSON(payload, js, contents);
        TO_JSON(payload, js, atomRadiusMultiplier);
        TO_JSON(payload, js, loadBonds);
        TO_JSON(payload, js, loadNonPolymerChemicals);
        TO_JSON(payload, js, loadHydrogen);
        TO_JSON(payload, js, representation);
        TO_JSON(payload, js, chainIds);
        TO_JSON(payload, js, recenter);
        TO_JSON(payload, js, occurrences);
        TO_JSON(payload, js, allowedOccurrences);
        TO_JSON(payload, js, animationParams);
        TO_JSON(payload, js, position);
        TO_JSON(payload, js, rotation);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

bool from_json(SugarDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, pdbId);
        FROM_JSON(param, js, contents);
        FROM_JSON(param, js, proteinName);
        FROM_JSON(param, js, atomRadiusMultiplier);
        FROM_JSON(param, js, loadBonds);
        FROM_JSON(param, js, representation);
        FROM_JSON(param, js, recenter);
        FROM_JSON(param, js, chainIds);
        FROM_JSON(param, js, siteIndices);
        FROM_JSON(param, js, rotation);
        FROM_JSON(param, js, animationParams);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(EnzymeReactionDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, enzymeName);
        FROM_JSON(param, js, substrateNames);
        FROM_JSON(param, js, productNames);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(EnzymeReactionProgressDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, instanceId);
        FROM_JSON(param, js, progress);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AddGridDetails &param, const std::string &payload)
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
        FROM_JSON(param, js, showPlanes);
        FROM_JSON(param, js, showFullGrid);
        FROM_JSON(param, js, useColors);
        FROM_JSON(param, js, position);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AddSphereDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, position);
        FROM_JSON(param, js, radius);
        FROM_JSON(param, js, color);
        FROM_JSON(param, js, opacity);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AddBoundingBoxDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, bottomLeft);
        FROM_JSON(param, js, topRight);
        FROM_JSON(param, js, radius);
        FROM_JSON(param, js, color);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ModelIdDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
        FROM_JSON(param, js, maxNbInstances);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(MaterialsDetails &param, const std::string &payload)
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
        FROM_JSON(param, js, castUserData);
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

std::string to_json(const IdsDetails &param)
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

bool from_json(NameDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, name);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

std::string to_json(const NameDetails &param)
{
    try
    {
        nlohmann::json js;
        TO_JSON(param, js, name);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

// Fields
bool from_json(BuildFieldsDetails &param, const std::string &payload)
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

bool from_json(FileAccessDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, filename);
        FROM_JSON(param, js, lowBounds);
        FROM_JSON(param, js, highBounds);
        FROM_JSON(param, js, fileFormat);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(DatabaseAccessDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, brickId);
        FROM_JSON(param, js, lowBounds);
        FROM_JSON(param, js, highBounds);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ModelIdFileAccessDetails &param, const std::string &payload)
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

bool from_json(BuildPointCloudDetails &param, const std::string &payload)
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

bool from_json(ModelsVisibilityDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, visible);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ProteinInstanceTransformationDetails &param,
               const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, instanceIndex);
        FROM_JSON(param, js, position);
        FROM_JSON(param, js, rotation);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(InspectionDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, origin);
        FROM_JSON(param, js, direction);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

std::string to_json(const ProteinInspectionDetails &param)
{
    try
    {
        nlohmann::json js;
        TO_JSON(param, js, hit);
        TO_JSON(param, js, assemblyName);
        TO_JSON(param, js, proteinName);
        TO_JSON(param, js, modelId);
        TO_JSON(param, js, instanceId);
        TO_JSON(param, js, position);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}

bool from_json(VasculatureDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, populationName);
        FROM_JSON(param, js, useSdf);
        FROM_JSON(param, js, gids);
        FROM_JSON(param, js, quality);
        FROM_JSON(param, js, radiusMultiplier);
        FROM_JSON(param, js, sqlFilter);
        FROM_JSON(param, js, scale);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(VasculatureColorSchemeDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, colorScheme);
        FROM_JSON(param, js, palette);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(VasculatureReportDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, populationName);
        FROM_JSON(param, js, simulationReportId);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(VasculatureRadiusReportDetails &param,
               const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, populationName);
        FROM_JSON(param, js, simulationReportId);
        FROM_JSON(param, js, frame);
        FROM_JSON(param, js, amplitude);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AstrocytesDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, populationName);
        FROM_JSON(param, js, vasculaturePopulationName);
        FROM_JSON(param, js, loadSomas);
        FROM_JSON(param, js, loadDendrites);
        FROM_JSON(param, js, loadEndFeet);
        FROM_JSON(param, js, generateInternals);
        FROM_JSON(param, js, useSdf);
        FROM_JSON(param, js, geometryQuality);
        FROM_JSON(param, js, morphologyColorScheme);
        FROM_JSON(param, js, populationColorScheme);
        FROM_JSON(param, js, radiusMultiplier);
        FROM_JSON(param, js, sqlFilter);
        FROM_JSON(param, js, scale);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(NeuronsDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, populationName);
        FROM_JSON(param, js, loadSomas);
        FROM_JSON(param, js, loadAxon);
        FROM_JSON(param, js, loadBasalDendrites);
        FROM_JSON(param, js, loadApicalDendrites);
        FROM_JSON(param, js, loadSynapses);
        FROM_JSON(param, js, generateInternals);
        FROM_JSON(param, js, generateExternals);
        FROM_JSON(param, js, useSdf);
        FROM_JSON(param, js, geometryQuality);
        FROM_JSON(param, js, morphologyColorScheme);
        FROM_JSON(param, js, populationColorScheme);
        FROM_JSON(param, js, radiusMultiplier);
        FROM_JSON(param, js, sqlNodeFilter);
        FROM_JSON(param, js, sqlSectionFilter);
        FROM_JSON(param, js, scale);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(NeuronSectionDetails &param, const std::string &payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, assemblyName);
        FROM_JSON(param, js, neuronId);
        FROM_JSON(param, js, sectionId);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

std::string to_json(const NeuronSectionPointsDetails &param)
{
    try
    {
        nlohmann::json js;
        TO_JSON(param, js, status);
        TO_JSON(param, js, points);
        return js.dump();
    }
    catch (...)
    {
        return "";
    }
    return "";
}
#endif
