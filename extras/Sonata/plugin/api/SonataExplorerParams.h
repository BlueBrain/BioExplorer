/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include "../../common/Types.h"
#include <brayns/common/types.h>

namespace sonataexplorer
{
namespace api
{
struct Response
{
    bool status{true};
    std::string contents;
};
std::string to_json(const Response& param);

/** Save model to cache */
struct ExportModelToFile
{
    int32_t modelId;
    std::string path;
};

bool from_json(ExportModelToFile& modelSave, const std::string& payload);

/** Save model to mesh */
struct ExportModelToMesh
{
    int32_t modelId;
    std::string path;
    int32_t density;
    double radiusMultiplier;
    double shrinkFactor;
    bool skin;
};

bool from_json(ExportModelToMesh& modelSave, const std::string& payload);

struct MaterialDescriptor
{
    int32_t modelId;
    int32_t materialId;
    std::vector<float> diffuseColor;
    std::vector<float> specularColor;
    float specularExponent;
    float reflectionIndex;
    float opacity;
    float refractionIndex;
    float emission;
    float glossiness;
    bool simulationDataCast;
    int32_t shadingMode;
    int32_t clippingMode;
    float userParameter;
};

bool from_json(MaterialDescriptor& materialDescriptor,
               const std::string& payload);

struct MaterialsDescriptor
{
    std::vector<int32_t> modelIds;
    std::vector<int32_t> materialIds;
    std::vector<float> diffuseColors;
    std::vector<float> specularColors;
    std::vector<float> specularExponents;
    std::vector<float> reflectionIndices;
    std::vector<float> opacities;
    std::vector<float> refractionIndices;
    std::vector<float> emissions;
    std::vector<float> glossinesses;
    std::vector<bool> simulationDataCasts;
    std::vector<int32_t> shadingModes;
    std::vector<int32_t> clippingModes;
    std::vector<float> userParameters;
};

bool from_json(MaterialsDescriptor& materialsDescriptor,
               const std::string& payload);

struct MaterialRangeDescriptor
{
    int32_t modelId;
    std::vector<int32_t> materialIds;
    std::vector<float> diffuseColor;
    std::vector<float> specularColor;
    float specularExponent;
    float reflectionIndex;
    float opacity;
    float refractionIndex;
    float emission;
    float glossiness;
    bool simulationDataCast;
    int32_t shadingMode;
    int32_t clippingMode;
    float userParameter;
};

bool from_json(MaterialRangeDescriptor& materialRangeDescriptor,
               const std::string& payload);

// Material IDs for a given model
struct ModelId
{
    size_t modelId;
};

bool from_json(ModelId& modelId, const std::string& payload);

struct MaterialIds
{
    std::vector<size_t> ids;
};

std::string to_json(const MaterialIds& param);

/** Set extra attributes to materials */
struct MaterialExtraAttributes
{
    int32_t modelId;
};
bool from_json(MaterialExtraAttributes& param, const std::string& payload);

// Synapse attributes
struct SynapseAttributes
{
    std::string circuitConfiguration;
    int32_t gid;
    std::vector<std::string> htmlColors;
    float lightEmission;
    float radius;
};

bool from_json(SynapseAttributes& synapseAttributes,
               const std::string& payload);

/** Circuit bounding box */
struct CircuitBoundingBox
{
    std::vector<double> aabb{0, 0, 0, 0, 0, 0};
};

bool from_json(CircuitBoundingBox& circuitBoundingBox,
               const std::string& payload);

/** Connections per value */
struct ConnectionsPerValue
{
    int32_t modelId;
    int32_t frame;
    double value;
    double epsilon;
};

bool from_json(ConnectionsPerValue& connectionsPerValue,
               const std::string& payload);

struct AttachCellGrowthHandler
{
    uint64_t modelId;
    uint64_t nbFrames;
};
bool from_json(AttachCellGrowthHandler& param, const std::string& payload);

struct AttachCircuitSimulationHandler
{
    uint64_t modelId;
    std::string circuitConfiguration;
    std::string reportName;
    bool synchronousMode;
};
bool from_json(AttachCircuitSimulationHandler& param,
               const std::string& payload);

struct AddGrid
{
    float minValue;
    float maxValue;
    float steps;
    float radius;
    float planeOpacity;
    bool showAxis;
    bool useColors;
};
bool from_json(AddGrid& param, const std::string& payload);

struct AddColumn
{
    float radius;
};
bool from_json(AddColumn& param, const std::string& payload);

struct AddSphere
{
    std::string name;
    std::vector<float> center;
    float radius;
    std::vector<double> color;
};
bool from_json(AddSphere& param, const std::string& payload);

struct AddPill
{
    std::string name;
    std::string type;
    std::vector<float> p1;
    std::vector<float> p2;
    float radius1;
    float radius2;
    std::vector<double> color;
};
bool from_json(AddPill& param, const std::string& payload);

struct AddCylinder
{
    std::string name;
    std::vector<float> center;
    std::vector<float> up;
    float radius;
    std::vector<double> color;
};
bool from_json(AddCylinder& param, const std::string& payload);

struct AddBox
{
    std::string name;
    std::vector<float> minCorner;
    std::vector<float> maxCorner;
    std::vector<double> color;
};
bool from_json(AddBox& param, const std::string& payload);

struct ImportMorphology
{
    std::string connectionString;
    std::string schema;
    uint64_t guid;
    std::string filename;
};
bool from_json(ImportMorphology& param, const std::string& payload);

struct ImportVolume
{
    std::string connectionString;
    std::string schema;
    uint64_t guid;
    std::vector<uint16_t> dimensions;
    std::vector<float> spacing;
    std::string rawFilename;
};
bool from_json(ImportVolume& param, const std::string& payload);

struct ImportCompartmentSimulation
{
    std::string connectionString;
    std::string schema;
    std::string blueConfig;
    std::string reportName;
    uint64_t reportId;
};
bool from_json(ImportCompartmentSimulation& param, const std::string& payload);

struct ImportSynapses
{
    std::string connectionString;
    std::string schema;
    std::string blueConfig;
};
bool from_json(ImportSynapses& param, const std::string& payload);
} // namespace api
} // namespace sonataexplorer
