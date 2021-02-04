/* Copyright (c) 2020-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: cyrille.favreau@epfl.ch
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

#include "AdvancedMaterial.h"

#include <ospray/SDK/render/Renderer.h>

namespace bioexplorer
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

    float _timestamp{0.f};
    float _exposure{1.f};

    float _fogThickness{1e6f};
    float _fogStart{0.f};

    ospray::uint32 _maxBounces{3};
    ospray::uint32 _randomNumber{0};

    float _shadows{0.f};
    float _softShadows{0.f};
    ospray::uint32 _softShadowsSamples{0};

    float _giStrength{0.f};
    float _giDistance{1e6f};
    ospray::uint32 _giSamples{0};
};
} // namespace bioexplorer
