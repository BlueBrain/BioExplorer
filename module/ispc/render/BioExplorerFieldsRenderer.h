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

#pragma once

#include "BioExplorerMaterial.h"

#include <ospray/SDK/render/Renderer.h>

namespace brayns
{
class BioExplorerFieldsRenderer : public ospray::Renderer
{
public:
    BioExplorerFieldsRenderer();

    /**
       Returns the class name as a string
       @return string containing the full name of the class
    */
    std::string toString() const final { return "bio_explorer_fields"; }
    void commit() final;

private:
    // Shading attributes
    std::vector<void*> _lightArray;
    void** _lightPtr;
    ospray::Data* _lightData;

    brayns::obj::BioExplorerMaterial* _bgMaterial;

    bool _useHardwareRandomizer{false};
    ospray::uint32 _randomNumber{0};

    float _timestamp{0.f};
    float _exposure{1.f};

    float _shadows{0.f};
    float _softShadows{0.f};

    bool _shadingEnabled{false};
    bool _softnessEnabled{false};

    // Octree
    float _step;
    ospray::uint32 _maxSteps;
    float _cutoff;
    ospray::Ref<ospray::Data> _userData;
    ospray::uint64 _userDataSize;
};
} // namespace brayns
