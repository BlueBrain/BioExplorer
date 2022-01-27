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

#pragma once

#include "AdvancedMaterial.h"

#include <ospray/SDK/render/Renderer.h>

namespace bioexplorer
{
namespace rendering
{
/**
 * @brief The AdvancedRenderer class is a renderer that can
 * perform global illumination (light shading, shadows, ambient occlusion, color
 * bleeding, light emission)
 */
class AdvancedRenderer : public ospray::Renderer
{
public:
    /**
     * @brief Construct a new Bio Explorer Renderer object
     *
     */
    AdvancedRenderer();

    /**
     * @brief Returns the class name as a string
     *
     * @return A string containing the full name of the class
     */
    std::string toString() const final { return "bio_explorer_renderer"; }

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

    bool _useHardwareRandomizer{false};
    bool _showBackground{false};

    double _timestamp{0.f};
    double _exposure{1.f};

    double _fogThickness{1e6f};
    double _fogStart{0.f};

    ospray::uint32 _maxBounces{3};
    ospray::uint32 _randomNumber{0};

    double _shadows{0.f};
    double _softShadows{0.f};
    ospray::uint32 _softShadowsSamples{0};

    double _giStrength{0.f};
    double _giDistance{1e6f};
    ospray::uint32 _giSamples{0};

    bool _matrixFilter{false};

    ospray::Ref<ospray::Data> _simulationData;
    ospray::uint64 _simulationDataSize;
};
} // namespace rendering
} // namespace bioexplorer
