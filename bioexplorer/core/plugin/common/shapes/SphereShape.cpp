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

#include "SphereShape.h"

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

SphereShape::SphereShape(const Vector4ds& clippingPlanes, const double radius)
    : Shape(clippingPlanes)
    , _radius(radius)
{
    const auto r = radius / 2.0;
    _bounds.merge(Vector3d(-r, -r, -r));
    _bounds.merge(Vector3d(r, r, r));
    _surface = 4.0 * M_PI * _radius * _radius;
}

Transformation SphereShape::getTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const AnimationDetails& animationDetails, const double offset) const
{
    if (animationDetails.morphingStep == 0.f)
        return _getTransformation(occurrence, nbOccurrences, animationDetails,
                                  offset);
    else
        return _getMorphedTransformation(occurrence, nbOccurrences,
                                         animationDetails, offset);
}

Transformation SphereShape::_getTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const AnimationDetails& animationDetails, const double offset) const
{
    uint64_t occ = occurrence;
    uint64_t occs = nbOccurrences;
    if (occs != 0 && animationDetails.seed != 0)
    {
        occs = 36000;
        occ = rand() % occs;
    }

    const double radius =
        _radius + (animationDetails.positionSeed == 0
                       ? animationDetails.positionStrength
                       : animationDetails.positionStrength *
                             rnd3(animationDetails.positionSeed + occ));

    Vector3d pos;
    Quaterniond rot;
    sphereFilling(radius, occ, occs, pos, rot, offset);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    if (animationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, animationDetails.rotationSeed, occ,
                                     animationDetails.rotationStrength);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

Transformation SphereShape::_getMorphedTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const AnimationDetails& animationDetails, const double offset) const
{
    uint64_t rnd = occurrence;
    if (nbOccurrences != 0 && animationDetails.seed != 0)
        rnd = rand() % nbOccurrences;

    // Position randomizer
    double R = _radius + (animationDetails.positionSeed == 0
                              ? animationDetails.positionStrength
                              : animationDetails.positionStrength *
                                    rnd3(animationDetails.positionSeed + rnd));

    // Sphere filling
    const double off = 2.0 / nbOccurrences;
    const double increment = M_PI * (3.0 - sqrt(5.0));
    const double y = ((rnd * off) - 1.0) + (off / 2.0);
    const double r = sqrt(1.f - pow(y, 2.0));
    const double phi = rnd * increment;
    const double x = cos(phi) * r;
    const double z = sin(phi) * r;

    const Vector3d startDir{x, y, z};
    const Vector3d startPos = (R + offset) * startDir;

    if (isClipped(startPos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    Vector3d endPos = startPos;

    Quaterniond startRotation = safeQuatlookAt(startDir);
    if (animationDetails.rotationSeed != 0)
        startRotation =
            weightedRandomRotation(startRotation, animationDetails.rotationSeed,
                                   rnd, animationDetails.rotationStrength);

    R = _radius;
    const double endRadius = R * 2.f;

    endPos.y = -R;
    endPos = endPos + (1.f - (startPos.y + (R + offset)) / endRadius) *
                          Vector3d(endRadius, 0.f, endRadius) *
                          normalize(Vector3d(startDir.x, 0.f, startDir.z));

    const Quaterniond endRotation{0.f, 0.f, 0.707f, 0.707f};
    const Quaterniond finalRotation =
        glm::lerp(startRotation, endRotation,
                  double(animationDetails.morphingStep));

    // Final transformation
    Transformation transformation;
    const Vector3d finalTranslation =
        endPos * animationDetails.morphingStep +
        startPos * (1.f - animationDetails.morphingStep);
    transformation.setTranslation(finalTranslation);
    transformation.setRotation(finalRotation);
    return transformation;
}

bool SphereShape::isInside(const Vector3d& point) const
{
    return length(point) <= _radius;
}

} // namespace common
} // namespace bioexplorer
