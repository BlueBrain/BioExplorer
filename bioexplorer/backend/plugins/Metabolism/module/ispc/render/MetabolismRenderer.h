/*
 * Copyright 2020-2023 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <ospray/SDK/render/Renderer.h>

namespace metabolism
{
namespace rendering
{
/**
 * @brief The MetabolismRenderer class allows visualization of atom Metabolism
 * in the 3D scene
 */
class MetabolismRenderer : public ospray::Renderer
{
public:
    /**
     * @brief Construct a new Bio Explorer Metabolism Renderer object
     *
     */
    MetabolismRenderer();

    /**
     * @brief Returns the class name as a string
     *
     * @return A string containing the full name of the class
     */
    std::string toString() const final { return "bio_explorer_Metabolism"; }

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

    ospray::Material* _bgMaterial;

    float _exposure{1.f};

    float _nearPlane{100.f};
    float _farPlane{1.f};
    float _rayStep{0.1f};
    ospray::uint32 _refinementSteps;
    float _alphaCorrection{1.f};
    float _noiseFrequency{1.f};
    float _noiseAmplitude{1.f};
    bool _colorMapPerRegion{false};

    ospray::Ref<ospray::Data> _userData;
    ospray::uint64 _userDataSize;
};
} // namespace rendering
} // namespace metabolism
