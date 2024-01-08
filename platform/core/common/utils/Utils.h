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

#include <platform/core/common/MathTypes.h>
#include <platform/core/common/Types.h>

#include <algorithm>

namespace core
{
strings parseFolder(const std::string& folder, const strings& filters);

std::string extractExtension(const std::string& filename);

template <size_t M, typename T>
inline glm::vec<M, T> toGlmVec(const std::array<T, M>& input)
{
    glm::vec<M, T> vec;
    memcpy(glm::value_ptr(vec), input.data(), input.size() * sizeof(T));
    return vec;
}

template <size_t M, typename T>
inline std::array<T, M> toArray(const glm::vec<M, T>& input)
{
    std::array<T, M> output;
    memcpy(output.data(), glm::value_ptr(input), M * sizeof(T));
    return output;
}

/**
 * @brief Get the Bezier Point from a curve defined by the provided control
 * points
 *
 * @param controlPoints Curve control points with radius
 * @param t The t in the function for a curve can be thought of as describing
 * how far B(t) is from first to last control point.
 * @return Vector3f
 */
Vector4f getBezierPoint(const Vector4fs& controlPoints, const float t);

/**
 * @brief Get the Rainbow Colormap
 *
 * @param colormapSize Size of the colormap
 * @return Vector3fs RGB colors of the colormap
 */
Vector3fs getRainbowColormap(const uint32_t colormapSize);

/**
 * @brief Function template taking two template parameters (To and From) representing the source and target types
 *
 * @tparam To Source type
 * @tparam From Destination type
 * @param from Input data
 * @return To Output data
 */
template <typename To, typename From>
To lexical_cast(const From& from);

} // namespace core
