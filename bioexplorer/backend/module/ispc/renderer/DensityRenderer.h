/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

// Platform
#include <platform/engines/ospray/ispc/render/utils/AdvancedMaterial.h>

// OSPRay
#include <ospray/SDK/render/Renderer.h>

namespace bioexplorer
{
namespace rendering
{
using namespace ospray;
using namespace core;

/**
 * @brief The DensityRenderer class allows visualization of atom density in the
 * 3D scene
 */
class DensityRenderer : public ospray::Renderer
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
     * @return A string containing the full name of the class
     */
    std::string toString() const final { return "bio_explorer_density"; }

    /**
     * @brief Commit the changes to the OSPRay engine
     *
     */
    void commit() final;

private:
    // Shading attributes
    std::vector<void*> _lightArray;
    void** _lightPtr;
    ospray::Data* _lightData;

    AdvancedMaterial* _bgMaterial;

    double _timestamp{0.f};
    double _exposure{1.f};

    double _alphaCorrection{1.f};

    double _rayStep;
    double _searchLength;
    double _farPlane;
    ospray::uint32 _samplesPerFrame;
};
} // namespace rendering
} // namespace bioexplorer
