/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

SphereShape::SphereShape(const bool filled, const Vector4ds& clippingPlanes,
                         const double radius)
    : Shape(clippingPlanes)
    , _filled(filled)
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
    if (_filled)
        return _getFilledSphereTransformation(occurrence, nbOccurrences,
                                              animationDetails, offset);

    if (animationDetails.morphingStep == 0.f)
        return _getEmptySphereTransformation(occurrence, nbOccurrences,
                                             animationDetails, offset);
    else
        return _getEmptySphereMorphedTransformation(occurrence, nbOccurrences,
                                                    animationDetails, offset);
}

Transformation SphereShape::_getEmptySphereTransformation(
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
    sphereFilling(radius + offset, occurrence, nbOccurrences, rnd, pos, rot);

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

Transformation SphereShape::_getEmptySphereMorphedTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const AnimationDetails& animationDetails, const double offset) const
{
    uint64_t rnd = occurrence;
    if (nbOccurrences != 0 && animationDetails.seed != 0)
        rnd = rand() % nbOccurrences;

    const double radius =
        _radius + (animationDetails.positionSeed == 0
                       ? animationDetails.positionStrength
                       : animationDetails.positionStrength *
                             rnd3(animationDetails.positionSeed + rnd));

    Vector3d startPos;
    Quaterniond startRot;
    const Vector3d startDir = sphereFilling(radius, occurrence, nbOccurrences,
                                            rnd, startPos, startRot, offset);

    if (animationDetails.rotationSeed != 0)
        startRot =
            weightedRandomRotation(startRot, animationDetails.rotationSeed, rnd,
                                   animationDetails.rotationStrength);

    const double endRadius = radius * 2.0;
    const double morphingStep = animationDetails.morphingStep;

    Vector3d endPos = startPos;
    endPos.y = -radius;
    endPos = endPos + (1.0 - (startPos.y + _radius) / endRadius) *
                          Vector3d(endRadius, 0.0, endRadius) *
                          normalize(Vector3d(startDir.x, 0.0, startDir.z));

    Quaterniond endRot{0.0, 0.0, -0.707, 0.707};
    if (animationDetails.rotationSeed != 0)
        endRot = weightedRandomRotation(endRot, animationDetails.rotationSeed,
                                        rnd, animationDetails.rotationStrength);

    const Quaterniond finalRotation = slerp(startRot, endRot, morphingStep);

    const auto finalTranslation =
        endPos * morphingStep + startPos * (1.0 - morphingStep);
    if (isClipped(finalTranslation, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    // Final transformation
    Transformation transformation;
    transformation.setTranslation(finalTranslation);
    transformation.setRotation(finalRotation);
    return transformation;
}

bool SphereShape::isInside(const Vector3d& point) const
{
    return length(point) <= _radius;
}

Transformation SphereShape::_getFilledSphereTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const AnimationDetails& animationDetails, const double offset) const
{
    Vector3d pos;
    const double diameter = _radius * 2.0;
    do
    {
        pos = Vector3d(rnd1() * diameter, rnd1() * diameter, rnd1() * diameter);
    } while (length(pos) > _radius);

    if (animationDetails.positionSeed != 0)
    {
        const Vector3d posOffset =
            animationDetails.positionStrength *
            Vector3d(rnd2(occurrence + animationDetails.positionSeed),
                     rnd2(occurrence + animationDetails.positionSeed + 1),
                     rnd2(occurrence + animationDetails.positionSeed + 2));

        pos += posOffset;
    }
    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    Quaterniond rot = safeQuatlookAt(normalize(pos));
    if (animationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, animationDetails.rotationSeed,
                                     occurrence,
                                     animationDetails.rotationStrength);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

} // namespace common
} // namespace bioexplorer
