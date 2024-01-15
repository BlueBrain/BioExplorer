/*
 * Copyright (c) 2015-2024, EPFL/Blue Brain Project
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include <ospray.h>
#include <platform/core/engineapi/Field.h>

namespace core
{
namespace engine
{
namespace ospray
{
class OSPRayField : public Field
{
public:
    OSPRayField(const FieldParameters& parameters, const Vector3ui& dimensions, const Vector3f& spacing,
                const Vector3f& offset, const uint32_ts& indices, const floats& values, const OctreeDataType dataType)
        : Field(parameters, dimensions, spacing, offset, indices, values, dataType)
    {
    }
};
} // namespace ospray
} // namespace engine
} // namespace core
