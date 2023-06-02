/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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
#include <plugin/common/Utils.h>

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
