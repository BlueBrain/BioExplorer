/* Copyright (c) 2018, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Jonas Karlsson <jonas.karlsson@epfl.ch>
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

#include <core/brayns/common/Types.h>

namespace brayns
{
enum class CurveType
{
    flat = 0,
    round = 1,
    ribbon = 2
};

enum class BaseType
{
    linear = 0,
    bezier = 1,
    bspline = 2,
    hermite = 3
};

inline std::string baseTypeAsString(const BaseType baseType)
{
    const strings values = {"linear", "bezier", "bspline", "hermite"};
    return values[static_cast<size_t>(baseType)];
};

inline std::string curveTypeAsString(const CurveType curveType)
{
    const strings values = {"flat", "round", "ribbon"};
    return values[static_cast<size_t>(curveType)];
};

struct Curve
{
    Curve(const CurveType curveType_in, const BaseType baseType_in, const Vector4fs& vertices_in,
          const uint64_ts& indices_in, const Vector3fs& normals_in, const Vector3fs& tangents_in)
        : curveType(curveType_in)
        , baseType(baseType_in)
        , vertices(vertices_in)
        , indices(indices_in)
        , normals(normals_in)
        , tangents(tangents_in)
    {
    }

    // Curve type: falt, round, ribbon
    CurveType curveType;

    // Base type: linear, bezier, bspline, hermite
    BaseType baseType;

    // Vertices
    Vector4fs vertices;

    // Indices
    uint64_ts indices;

    // Normals (for ribbon curves)
    Vector3fs normals;

    // Tangents (for hermite curves)
    Vector3fs tangents;
};
} // namespace brayns
