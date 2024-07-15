/*
    Copyright 2018 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <platform/core/common/Types.h>

namespace core
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
} // namespace core
