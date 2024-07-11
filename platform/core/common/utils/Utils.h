/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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
