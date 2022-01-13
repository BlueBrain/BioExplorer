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

#include "FanShape.h"

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

FanShape::FanShape(const Vector4ds& clippingPlanes, const double radius)
    : Shape(clippingPlanes)
    , _radius(radius)
{
    const auto r = radius / 2.f;
    _bounds.merge(Vector3d(-r, -r, -r));
    _bounds.merge(Vector3d(r, r, r));
    _surface = 4.f * M_PI * _radius * _radius;
}

Transformation FanShape::getTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const AnimationDetails& animationDetails, const double offset) const
{
    uint64_t rnd = occurrence;
    if (nbOccurrences != 0 && animationDetails.seed != 0)
        if (GeneralSettings::getInstance()->getV1Compatibility())
            rnd = rand() % nbOccurrences;
        else
            rnd = rand() % std::numeric_limits<uint64_t>::max();

    const double radius =
        _radius + (animationDetails.positionSeed == 0
                       ? animationDetails.positionStrength
                       : animationDetails.positionStrength *
                             rnd3(animationDetails.positionSeed + rnd));

    Vector3d pos;
    Quaterniond rot;
    sphereFilling(radius, occurrence, nbOccurrences, rnd, pos, rot, offset,
                  0.1);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    if (animationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, animationDetails.rotationSeed, rnd,
                                     animationDetails.rotationStrength);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool FanShape::isInside(const Vector3d& point) const
{
    PLUGIN_THROW("isInside is not implemented for Fan shapes");
}

} // namespace common
} // namespace bioexplorer
