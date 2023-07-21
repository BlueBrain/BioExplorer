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

#include <platform/engines/ospray/ispc/render/utils/AbstractRenderer.h>

namespace spaceexplorer
{
namespace blackhole
{
class BlackHoleRenderer : public core::AbstractRenderer
{
public:
    BlackHoleRenderer();

    /**
       Returns the class name as a string
       @return string containing the full name of the class
    */
    std::string toString() const final { return "blackhole"; }
    void commit() final;

private:
    // Shading attributes
    float _exposure{1.f};
    ospray::uint32 _nbDisks;
    bool _grid{false};
    float _diskRotationSpeed{3.0};
    ospray::uint32 _diskTextureLayers{12};
    float _blackHoleSize{0.3f};
};
} // namespace blackhole
} // namespace spaceexplorer
