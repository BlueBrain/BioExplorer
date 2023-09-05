/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include "SDFGeometries.h"

#include <science/common/UniqueId.h>
#include <science/common/Utils.h>

#include <platform/core/common/utils/Utils.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>

using namespace core;

namespace bioexplorer
{
using namespace morphology;

namespace common
{
SDFGeometries::SDFGeometries(const double alignToGrid, const Vector3d& position, const Quaterniond& rotation,
                             const Vector3d& scale)
    : Node(scale)
    , _alignToGrid(alignToGrid)
    , _position(position)
    , _rotation(rotation)
{
}

Vector3d SDFGeometries::_animatedPosition(const Vector4d& position, const uint64_t index) const
{
    if (_animationDetails.seed == 0)
        return Vector3d(position);
    const auto seed = _animationDetails.seed + _animationDetails.offset * index;
    const auto amplitude = _animationDetails.amplitude * position.w;
    const auto frequency = _animationDetails.frequency;
    return Vector3d(position.x + amplitude * rnd3(seed + position.x * frequency),
                    position.y + amplitude * rnd3(seed + position.y * frequency),
                    position.z + amplitude * rnd3(seed + position.z * amplitude));
}

Vector4fs SDFGeometries::_getProcessedSectionPoints(const MorphologyRepresentation& representation,
                                                    const Vector4fs& points)
{
    Vector4fs localPoints;
    if (representation == MorphologyRepresentation::bezier && points.size() > DEFAULT_BEZIER_STEP * 2)
    {
        for (double t = 0.0; t <= 1.0; t += 1.0 / static_cast<double>(points.size() * DEFAULT_BEZIER_STEP))
            localPoints.push_back(getBezierPoint(points, t));
    }
    else
        localPoints = points;
    return localPoints;
}

double SDFGeometries::_getCorrectedRadius(const double radius, const double radiusMultiplier) const
{
    if (radiusMultiplier < 0.0)
        return -radiusMultiplier;
    return radius * radiusMultiplier;
}

} // namespace common
} // namespace bioexplorer
