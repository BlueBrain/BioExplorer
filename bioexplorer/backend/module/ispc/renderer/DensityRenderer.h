/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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

#include <science/common/Properties.h>

#include <platform/engines/ospray/ispc/render/utils/AbstractRenderer.h>

namespace bioexplorer
{
namespace rendering
{

/**
 * @brief The DensityRenderer class allows visualization of atom density in the
 * 3D scene
 */
class DensityRenderer : public ::core::engine::ospray::AbstractRenderer
{
public:
    /**
     * @brief Construct a new Bio Explorer Density Renderer object
     *
     */
    DensityRenderer();

    /**
     * @brief Returns the class name as a string
     *
     * @return A string containing the name of the object in the OSPRay context
     */
    std::string toString() const final { return RENDERER_DENSITY; }

    /**
     * @brief Commit the changes to the OSPRay engine
     *
     */
    void commit() final;

private:
    // Shading attributes
    double _exposure{1.f};

    double _alphaCorrection{1.f};

    double _rayStep;
    double _searchLength;
    double _farPlane;
    ::ospray::uint32 _samplesPerFrame;
};
} // namespace rendering
} // namespace bioexplorer
