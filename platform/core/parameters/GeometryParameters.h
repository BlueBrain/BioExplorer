/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include "AbstractParameters.h"

#include <platform/core/common/Types.h>

SERIALIZATION_ACCESS(GeometryParameters)

namespace core
{
/** Manages geometry parameters
 */
class GeometryParameters : public AbstractParameters
{
public:
    /**
       Parse the command line parameters and populates according class members
     */
    GeometryParameters();

    /** @copydoc AbstractParameters::print */
    void print() final;

    /**
     * Defines what memory mode should be used between Core and the
     * underlying renderer
     */
    MemoryMode getMemoryMode() const { return _memoryMode; };
    const std::set<BVHFlag>& getDefaultBVHFlags() const { return _defaultBVHFlags; }

    /**
     * @brief Get the geometry quality (low, medium or high)
     *
     * @return GeometryQuality Geometry quality level
     */
    GeometryQuality getGeometryQuality() const { return _geometryQuality; }

    /**
     * @brief Get the epsilon for SDF geometries
     *
     * @return float Epsilon
     */
    float getSdfEpsilon() const { return _sdfEpsilon; }

    /**
     * @brief Get the blending factor for SDF geometries
     *
     * @return float Blending factor
     */
    float getSdfBlendFactor() const { return _sdfBlendFactor; }

    /**
     * @brief Get the blending lerp factor for SDF geometries
     *
     * @return float Blending lerp factor
     */
    float getSdfBlendLerpFactor() const { return _sdfBlendLerpFactor; }

    /**
     * @brief Get the number of ray-marching iterations for SDF geometries
     *
     * @return float Number of ray-marching iterations
     */
    float getSdfNbMarchIterations() const { return _sdfNbMarchIterations; }

    /**
     * @brief Get the ray-marching omega for SDF geometries
     *
     * @return float Value of Omega
     */
    float getSdfOmega() const { return _sdfOmega; }

    /**
     * @brief Get the distance until which SDF geometries are processed (blending and displacement)
     *
     * @return float The distance
     */
    float getSdfDistance() const { return _sdfDistance; }

protected:
    void parse(const po::variables_map& vm) final;

    // Scene
    std::set<BVHFlag> _defaultBVHFlags;

    // Geometry
    GeometryQuality _geometryQuality{GeometryQuality::high};
    float _sdfEpsilon{DEFAULT_GEOMETRY_SDF_EPSILON};
    uint64_t _sdfNbMarchIterations{DEFAULT_GEOMETRY_SDF_NB_MARCH_ITERATIONS};
    float _sdfBlendFactor{DEFAULT_GEOMETRY_SDF_BLEND_FACTOR};
    float _sdfBlendLerpFactor{DEFAULT_GEOMETRY_SDF_BLEND_LERP_FACTOR};
    float _sdfOmega{DEFAULT_GEOMETRY_SDF_OMEGA};
    float _sdfDistance{DEFAULT_GEOMETRY_SDF_DISTANCE};

    // System parameters
    MemoryMode _memoryMode{MemoryMode::shared};

    SERIALIZATION_FRIEND(GeometryParameters)
};
} // namespace core
