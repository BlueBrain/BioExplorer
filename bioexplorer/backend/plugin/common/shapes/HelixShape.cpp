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

#include "HelixShape.h"

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

HelixShape::HelixShape(const Vector4ds& clippingPlanes, const double radius, const double height)
    : Shape(clippingPlanes)
    , _height(height)
    , _radius(radius)
{
    _bounds.merge(Vector3d(0.0, 0.0, height) + Vector3d(radius, radius, radius));
    _bounds.merge(Vector3d(-radius, -radius, -radius));
    _surface = 2.0 * M_PI * _radius * _radius + _height * (2.0 * M_PI * _radius);
}

Transformation HelixShape::getTransformation(const uint64_t occurrence, const uint64_t nbOccurrences,
                                             const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
                                             const double offset) const
{
    const double radius = _radius + (MolecularSystemAnimationDetails.positionSeed == 0
                                         ? MolecularSystemAnimationDetails.positionStrength
                                         : MolecularSystemAnimationDetails.positionStrength *
                                               rnd3(MolecularSystemAnimationDetails.positionSeed + occurrence));

    const Vector3d pos =
        Vector3d(radius * cos(occurrence), radius * sin(occurrence), -_height * occurrence / nbOccurrences);
    const Vector3d normal = normalize(Vector3d(pos.x, pos.y, 0.0));
    Quaterniond rot = safeQuatlookAt(normal);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    if (MolecularSystemAnimationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, MolecularSystemAnimationDetails.rotationSeed, occurrence,
                                     MolecularSystemAnimationDetails.rotationStrength);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool HelixShape::isInside(const Vector3d& point) const
{
    return length(point) <= _radius;
}

} // namespace common
} // namespace bioexplorer
