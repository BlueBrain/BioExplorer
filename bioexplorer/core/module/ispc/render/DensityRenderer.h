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

#pragma once

#include "AdvancedMaterial.h"

#include <ospray/SDK/render/Renderer.h>

namespace bioexplorer
{
namespace rendering
{
using namespace ospray;

/**
 * @brief The DensityRenderer class allows visualization of magnetic
 * fields created by atoms in the 3D scene. An Octree acceleration structure has
 * to be built by the be_build_fields API in order to feed the renderer with the
 * information needed to compute the value of the field for every point in the
 * 3D space
 */
class DensityRenderer : public ospray::Renderer
{
public:
    /**
     * @brief Construct a new Bio Explorer Fields Renderer object
     *
     */
    DensityRenderer();

    /**
     * @brief Returns the class name as a string
     *
     * @return A string containing the full name of the class
     */
    std::string toString() const final { return "bio_explorer_fields"; }

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

    float _timestamp{0.f};
    float _exposure{1.f};

    float _alphaCorrection{1.f};

    float _rayStep;
    float _searchLength;
    float _farPlane;
    ospray::uint32 _sampleCount;
};
} // namespace rendering
} // namespace bioexplorer
