/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

class GolgiStyleRenderer : public Renderer
{
public:
    GolgiStyleRenderer();

    /**
       Returns the class name as a string
       @return string containing the full name of the class
    */
    std::string toString() const final { return "core::GolgiStyleRenderer"; }
    void commit() final;

private:
    AdvancedMaterial* _bgMaterial{nullptr};
    float _exponent{5.f};
    bool _inverse{false};
};
} // namespace rendering
} // namespace bioexplorer
