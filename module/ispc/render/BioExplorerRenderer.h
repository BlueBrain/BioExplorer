/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#ifndef BIOEXPLORER_RENDERER_H
#define BIOEXPLORER_RENDERER_H

#include "BioExplorerMaterial.h"

#include <ospray/SDK/render/Renderer.h>

namespace BioExplorer
{
/**
 * @brief The BioExplorerRenderer class is a renderer that can
 * perform global illumination (light shading, shadows, ambient occlusion, color
 * bleeding, light emission)
 */
class BioExplorerRenderer : public ospray::Renderer
{
public:
    BioExplorerRenderer();

    /**
       @return string containing the full name of the class
    */
    std::string toString() const final { return "bio_explorer_renderer"; }
    void commit() final;

private:
    bool _useHardwareRandomizer;
    bool _showBackground;

    std::vector<void*> _lightArray;
    void** _lightPtr;
    ospray::Data* _lightData;

    brayns::obj::BioExplorerMaterial* _bgMaterial;

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
} // namespace BioExplorer

#endif // BIOEXPLORER_RENDERER_H
