/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <ospray/SDK/geometry/Geometry.h>

namespace core
{
namespace engine
{
namespace ospray
{
struct Fields : public ::ospray::Geometry
{
public:
    Fields();

    std::string toString() const final { return ("field"); }
    void finalize(::ospray::Model* model) final;
    void commit() final;

protected:
    ::ospray::Ref<::ospray::Data> _indices;
    ::ospray::Ref<::ospray::Data> _values;
    int _dataType;
    ::ospray::vec3i _dimensions;
    ::ospray::vec3f _spacing;
    ::ospray::vec3f _offset;
    float _distance;
    float _cutoff;
    float _gradientOffset;
    bool _gradientShadingEnabled;
    float _samplingRate;
    float _epsilon;
    int _accumulationSteps;
};
} // namespace ospray
} // namespace engine
} // namespace core
