/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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
