/*
 * Copyright (c) 2015-2023, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Jonas Karlsson <jonas.karlsson@epfl.ch>
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

#include "ospray/SDK/geometry/Geometry.h"

namespace core
{
namespace engine
{
namespace ospray
{
struct SDFGeometries : public ::ospray::Geometry
{
    std::string toString() const final { return "SDFGeometries"; }
    void finalize(::ospray::Model* model) final;

    ::ospray::Ref<::ospray::Data> data;
    ::ospray::Ref<::ospray::Data> neighbours;
    ::ospray::Ref<::ospray::Data> geometries;

    SDFGeometries();

private:
    std::vector<void*> ispcMaterials_;
};
} // namespace ospray
} // namespace engine
} // namespace core