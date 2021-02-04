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
using namespace ospray;

/**
 * @brief The FieldsRenderer class allows visualization of magnetic
 * fields created by atoms in the 3D scene. An Octree acceleration structure has
 * to be built by the be_build_fields API in order to feed the renderer with the
 * information needed to compute the value of the field for every point in the
 * 3D space
 */
class FieldsRenderer : public ospray::Renderer
{
public:
    /**
     * @brief Construct a new Bio Explorer Fields Renderer object
     *
     */
    FieldsRenderer();

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

    bool _useHardwareRandomizer{false};
    ospray::uint32 _randomNumber{0};

    float _timestamp{0.f};
    float _exposure{1.f};

    float _alphaCorrection{1.f};

    // Octree
    float _minRayStep;
    ospray::uint32 _nbRaySteps;
    ospray::uint32 _nbRayRefinementSteps;

    float _cutoff;
    ospray::Ref<ospray::Data> _userData;
    ospray::uint64 _userDataSize;
};
} // namespace bioexplorer
