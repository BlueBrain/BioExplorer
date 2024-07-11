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

#include "TransferFunction.h"

#include <platform/core/common/Logs.h>

#include <algorithm>

namespace
{
double _interpolatedOpacity(const core::Vector2ds& controlPointsSorted, const double x)
{
    const auto& firstPoint = controlPointsSorted.front();
    if (x <= firstPoint.x)
        return firstPoint.y;

    for (size_t i = 1; i < controlPointsSorted.size(); ++i)
    {
        const auto& current = controlPointsSorted[i];
        const auto& previous = controlPointsSorted[i - 1];
        if (x <= current.x)
        {
            const auto t = (x - previous.x) / (current.x - previous.x);
            return (1.0 - t) * previous.y + t * current.y;
        }
    }

    const auto& lastPoint = controlPointsSorted.back();
    return lastPoint.y;
}
} // namespace

namespace core
{
bool ColorMap::operator==(const ColorMap& rhs) const
{
    if (this == &rhs)
        return true;
    return name == rhs.name && colors == rhs.colors;
}

void ColorMap::clear()
{
    colors = {{0, 0, 0}, {1, 1, 1}};
}

TransferFunction::TransferFunction()
{
    clear();
}

void TransferFunction::clear()
{
    _colorMap.clear();
    _controlPoints = {{0, 0}, {1, 1}};
    _valuesRange = {0, 255};
    markModified();
}

floats TransferFunction::calculateInterpolatedOpacities() const
{
    constexpr size_t numSamples = 256;
    constexpr double dx = 1. / (numSamples - 1);

    auto tfPoints = getControlPoints();
    std::sort(tfPoints.begin(), tfPoints.end(), [](auto a, auto b) { return a.x < b.x; });

    floats opacities;
    opacities.reserve(numSamples);
    for (size_t i = 0; i < numSamples; ++i)
        opacities.push_back(_interpolatedOpacity(tfPoints, i * dx));
    return opacities;
}
} // namespace core
