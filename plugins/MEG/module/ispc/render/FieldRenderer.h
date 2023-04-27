/* Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Brayns <https://github.com/BlueBrain/Brayns>
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

#include <ospray/SDK/render/Renderer.h>

namespace brayns
{
class FieldRenderer : public ospray::Renderer
{
public:
    FieldRenderer();

    /**
       Returns the class name as a string
       @return string containing the full name of the class
    */
    std::string toString() const final { return "meg_fields"; }
    void commit() final;

private:
    // Shading attributes
    std::vector<void*> _lightArray;
    void** _lightPtr;
    ospray::Data* _lightData;

    ospray::uint32 _randomNumber{0};

    float _timestamp{0.f};
    float _exposure{1.f};
    float _cutoff{100.f};
    float _spaceDistortion{0.f};
    bool _renderDirections{false};
    bool _processGeometry{false};
    bool _normalized{false};

    float _alphaCorrection{1.f};

    ospray::uint32 _nbRaySteps;
    ospray::uint32 _nbRefinementSteps;

    // Data
    ospray::Ref<ospray::Data> _userData;
    ospray::uint64 _userDataSize;
};
} // namespace brayns