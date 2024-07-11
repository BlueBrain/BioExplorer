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

#include "SonataExplorerParams.h"
#include <plugin/json.hpp>

#include <common/Logs.h>

namespace sonataexplorer
{
namespace api
{
#ifndef PLATFORM_DEBUG_JSON_ENABLED
#define FROM_JSON(PARAM, JSON, NAME) PARAM.NAME = JSON[#NAME].get<decltype(PARAM.NAME)>()
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

std::string to_json(const Response& param)
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

bool from_json(ExportModelToFile& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
        FROM_JSON(param, js, path);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ExportModelToMesh& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
        FROM_JSON(param, js, path);
        FROM_JSON(param, js, density);
        FROM_JSON(param, js, radiusMultiplier);
        FROM_JSON(param, js, shrinkFactor);
        FROM_JSON(param, js, skin);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(MaterialDescriptor& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
        FROM_JSON(param, js, materialId);
        FROM_JSON(param, js, diffuseColor);
        FROM_JSON(param, js, specularColor);
        FROM_JSON(param, js, specularExponent);
        FROM_JSON(param, js, reflectionIndex);
        FROM_JSON(param, js, opacity);
        FROM_JSON(param, js, refractionIndex);
        FROM_JSON(param, js, emission);
        FROM_JSON(param, js, glossiness);
        FROM_JSON(param, js, simulationDataCast);
        FROM_JSON(param, js, shadingMode);
        FROM_JSON(param, js, clippingMode);
        FROM_JSON(param, js, userParameter);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(MaterialsDescriptor& param, const std::string& payload)
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
        FROM_JSON(param, js, simulationDataCasts);
        FROM_JSON(param, js, shadingModes);
        FROM_JSON(param, js, clippingModes);
        FROM_JSON(param, js, userParameters);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(MaterialRangeDescriptor& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
        FROM_JSON(param, js, materialIds);
        FROM_JSON(param, js, diffuseColor);
        FROM_JSON(param, js, specularColor);
        FROM_JSON(param, js, specularExponent);
        FROM_JSON(param, js, reflectionIndex);
        FROM_JSON(param, js, opacity);
        FROM_JSON(param, js, refractionIndex);
        FROM_JSON(param, js, emission);
        FROM_JSON(param, js, glossiness);
        FROM_JSON(param, js, simulationDataCast);
        FROM_JSON(param, js, shadingMode);
        FROM_JSON(param, js, clippingMode);
        FROM_JSON(param, js, userParameter);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ModelId& param, const std::string& payload)
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

std::string to_json(const MaterialIds& param)
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

bool from_json(MaterialExtraAttributes& param, const std::string& payload)
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

bool from_json(SynapseAttributes& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, circuitConfiguration);
        FROM_JSON(param, js, gid);
        FROM_JSON(param, js, htmlColors);
        FROM_JSON(param, js, lightEmission);
        FROM_JSON(param, js, radius);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(CircuitBoundingBox& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, aabb);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ConnectionsPerValue& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
        FROM_JSON(param, js, frame);
        FROM_JSON(param, js, value);
        FROM_JSON(param, js, epsilon);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AttachCellGrowthHandler& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
        FROM_JSON(param, js, nbFrames);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AttachCircuitSimulationHandler& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
        FROM_JSON(param, js, circuitConfiguration);
        FROM_JSON(param, js, reportName);
        FROM_JSON(param, js, synchronousMode);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AddGrid& param, const std::string& payload)
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
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AddColumn& param, const std::string& payload)
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

bool from_json(AddSphere& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, center);
        FROM_JSON(param, js, radius);
        FROM_JSON(param, js, color);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AddPill& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, type);
        FROM_JSON(param, js, p1);
        FROM_JSON(param, js, p2);
        FROM_JSON(param, js, radius1);
        FROM_JSON(param, js, radius2);
        FROM_JSON(param, js, color);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AddCylinder& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, center);
        FROM_JSON(param, js, up);
        FROM_JSON(param, js, radius);
        FROM_JSON(param, js, color);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(AddBox& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, minCorner);
        FROM_JSON(param, js, maxCorner);
        FROM_JSON(param, js, color);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(SpikeReportVisualizationSettings& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, modelId);
        FROM_JSON(param, js, restVoltage);
        FROM_JSON(param, js, spikingVoltage);
        FROM_JSON(param, js, timeInterval);
        FROM_JSON(param, js, decaySpeed);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(LoadMEGSettings& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, name);
        FROM_JSON(param, js, path);
        FROM_JSON(param, js, reportName);
        FROM_JSON(param, js, density);
        FROM_JSON(param, js, voxelSize);
        FROM_JSON(param, js, synchronous);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(EnableMorphologyCache& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, enabled);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

bool from_json(ImportCircuitMorphologies& param, const std::string& payload)
{
    try
    {
        auto js = nlohmann::json::parse(payload);
        FROM_JSON(param, js, circuitPath);
        FROM_JSON(param, js, populationName);
        FROM_JSON(param, js, morphologyPath);
    }
    catch (...)
    {
        return false;
    }
    return true;
}

} // namespace api
} // namespace sonataexplorer
