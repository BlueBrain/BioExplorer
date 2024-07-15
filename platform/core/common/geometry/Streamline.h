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
struct Streamline
{
    Streamline(const Vector3fs& position_in, const Vector4fs& color_in, const std::vector<float>& radius_in)
        : position(position_in)
        , color(color_in)
        , radius(radius_in)
    {
    }

    // Array of vertex positions
    Vector3fs position;

    // Array of corresponding vertex colors (RGBA)
    Vector4fs color;

    // Array of vertex radii
    std::vector<float> radius;
};

struct StreamlinesData
{
    // Data array of all vertex position (and optional radius) for all
    // streamlines
    Vector4fs vertex;

    // Data array of corresponding vertex colors (RGBA)
    Vector4fs vertexColor;

    // Data array of indices to the first vertex of a link.
    //
    // A streamlines geometry can contain multiple disjoint streamlines, each streamline is specified as a list of
    // segments (or links) referenced via index: each entry e of the index array points the first vertex of a link
    // (vertex[index[e]]) and the second vertex of the link is implicitly the directly following one
    // (vertex[index[e]+1]). For example, two streamlines of vertices (A-B-C-D) and (E-F-G), respectively, would
    // internally correspond to five links (A-B, B-C, C-D, E-F, and F-G), and would be specified via an array of
    // vertices [A,B,C,D,E,F,G], plus an array of link indices [0,1,2,4,5].
    std::vector<int32_t> indices;

    void clear()
    {
        vertex.clear();
        vertexColor.clear();
        indices.clear();
    }
};
} // namespace core
