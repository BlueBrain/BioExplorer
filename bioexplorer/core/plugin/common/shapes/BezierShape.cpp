/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "BezierShape.h"

#include <plugin/common/Logs.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

BezierShape::BezierShape(const Vector4fs& clippingPlanes,
                         const Vector3fs points)
    : Shape(clippingPlanes)
    , _points(points)
{
    for (const auto& point : points)
        _bounds.merge(point);
}

Transformation BezierShape::getTransformation(
    const uint64_t occurence, const uint64_t nbOccurences,
    const RandomizationDetails& randDetails, const float offset) const
{
    Vector3fs bezierPoints = _points;
    size_t i = bezierPoints.size() - 1;
    while (i > 0)
    {
        for (size_t k = 0; k < i; ++k)
            bezierPoints[k] =
                bezierPoints[k] +
                occurence * (bezierPoints[k + 1] - bezierPoints[k]);
        --i;
    }
    const Vector3f normal =
        cross({0.f, 0.f, 1.f}, normalize(bezierPoints[1] - bezierPoints[0]));

    Vector3f pos = bezierPoints[0];
    const Quaterniond rot = quatLookAt(normal, UP_VECTOR);

    pos += normal * offset;

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

Transformation BezierShape::getTransformation(
    const uint64_t occurence, const uint64_t nbOccurences,
    const RandomizationDetails& randDetails, const float offset,
    const float /*morphingStep*/) const
{
    return getTransformation(occurence, nbOccurences, randDetails, offset);
}

bool BezierShape::isInside(const Vector3f& point) const
{
    PLUGIN_THROW("isInside is not implemented for Bezier shapes");
}

} // namespace common
} // namespace bioexplorer
