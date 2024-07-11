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

#include <platform/core/common/Types.h>

SERIALIZATION_ACCESS(TransferFunction)

namespace core
{
struct ColorMap
{
    std::string name;
    Vector3fs colors;

    bool operator==(const ColorMap& rhs) const;

    void clear();
};

class TransferFunction : public BaseObject
{
public:
    TransferFunction();

    /** Reset to gray-scale with opacity [0..1] and value range [0,255]. */
    void clear();

    const Vector2ds& getControlPoints() const { return _controlPoints; }
    void setControlPoints(const Vector2ds& controlPoints) { _updateValue(_controlPoints, controlPoints); }

    const ColorMap& getColorMap() const { return _colorMap; }
    void setColorMap(const ColorMap& colorMap) { _updateValue(_colorMap, colorMap); }

    const auto& getColors() const { return _colorMap.colors; }
    const Vector2d& getValuesRange() const { return _valuesRange; }
    void setValuesRange(const Vector2d& valuesRange) { _updateValue(_valuesRange, valuesRange); }

    floats calculateInterpolatedOpacities() const;

private:
    ColorMap _colorMap;
    Vector2ds _controlPoints;
    Vector2d _valuesRange;

    SERIALIZATION_FRIEND(TransferFunction)
};
} // namespace core
