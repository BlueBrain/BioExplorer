/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

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

#include "BezierShape.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace core;
using namespace details;

BezierShape::BezierShape(const Vector4ds& clippingPlanes, const Vector3ds points)
    : Shape(clippingPlanes)
{
    for (const auto& point : points)
    {
        _points.push_back(point);
        _bounds.merge(point);
    }
}

Transformation BezierShape::getTransformation(const uint64_t occurrence, const uint64_t nbOccurrences,
                                              const MolecularSystemAnimationDetails& molecularSystemAnimationDetails,
                                              const double offset) const
{
    Vector3ds bezierPoints = _points;
    size_t i = bezierPoints.size() - 1;
    while (i > 0)
    {
        for (size_t k = 0; k < i; ++k)
            bezierPoints[k] =
                bezierPoints[k] + static_cast<double>(occurrence) * (bezierPoints[k + 1] - bezierPoints[k]);
        --i;
    }
    const Vector3d normal = cross({0.0, 0.0, 1.0}, normalize(bezierPoints[1] - bezierPoints[0]));

    Vector3d pos = bezierPoints[0];

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    const Quaterniond rot = safeQuatlookAt(normal);

    pos += normal * offset;

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool BezierShape::isInside(const Vector3d& point) const
{
    PLUGIN_THROW("isInside is not implemented for Bezier shapes");
}

} // namespace common
} // namespace bioexplorer
