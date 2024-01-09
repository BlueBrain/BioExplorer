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

#include "ospray/SDK/common/Model.h"
#include "ospray/SDK/volume/Volume.h"

namespace core
{
namespace engine
{
namespace ospray
{
struct Field : public ::ospray::Volume
{
public:
    Field();

    std::string toString() const final { return ("octree_volume"); }
    void commit() final;
    int setRegion(const void *source, const ::ospray::vec3i &index, const ::ospray::vec3i &count) final;

protected:
    ::ospray::Ref<::ospray::Data> _indices;
    ::ospray::Ref<::ospray::Data> _values;
    ::ospray::vec3i _dimensions;
    ::ospray::vec3f _spacing;
    ::ospray::vec3f _offset;
};
} // namespace ospray
} // namespace engine
} // namespace core
