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

#include "SphereShape.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace core;
using namespace details;

SphereShape::SphereShape(const bool filled, const Vector4ds& clippingPlanes, const double radius)
    : Shape(clippingPlanes)
    , _filled(filled)
    , _radius(radius)
{
    const auto r = radius / 2.0;
    _bounds.merge(Vector3d(-r, -r, -r));
    _bounds.merge(Vector3d(r, r, r));
    _surface = 4.0 * M_PI * _radius * _radius;
}

Transformation SphereShape::getTransformation(const uint64_t occurrence, const uint64_t nbOccurrences,
                                              const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
                                              const double offset) const
{
    if (_filled)
        return _getFilledSphereTransformation(occurrence, nbOccurrences, MolecularSystemAnimationDetails, offset);

    if (MolecularSystemAnimationDetails.morphingStep == 0.f)
        return _getEmptySphereTransformation(occurrence, nbOccurrences, MolecularSystemAnimationDetails, offset);
    else
        return _getEmptySphereMorphedTransformation(occurrence, nbOccurrences, MolecularSystemAnimationDetails, offset);
}

Transformation SphereShape::_getEmptySphereTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails, const double offset) const
{
    uint64_t rnd = occurrence;
    if (nbOccurrences != 0 && MolecularSystemAnimationDetails.seed != 0)
    {
        if (GeneralSettings::getInstance()->getV1Compatibility())
            rnd = rand() % nbOccurrences;
        else
            rnd = rand() % std::numeric_limits<uint64_t>::max();
    }

    const double radius = _radius + (MolecularSystemAnimationDetails.positionSeed == 0
                                         ? MolecularSystemAnimationDetails.positionStrength
                                         : MolecularSystemAnimationDetails.positionStrength *
                                               rnd3(MolecularSystemAnimationDetails.positionSeed + rnd));

    Vector3d pos;
    Quaterniond rot;
    sphereFilling(radius + offset, occurrence, nbOccurrences, rnd, pos, rot);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    if (MolecularSystemAnimationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, MolecularSystemAnimationDetails.rotationSeed, rnd,
                                     MolecularSystemAnimationDetails.rotationStrength);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

Transformation SphereShape::_getEmptySphereMorphedTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails, const double offset) const
{
    uint64_t rnd = occurrence;
    if (nbOccurrences != 0 && MolecularSystemAnimationDetails.seed != 0)
        rnd = rand() % nbOccurrences;

    const double radius = _radius + (MolecularSystemAnimationDetails.positionSeed == 0
                                         ? MolecularSystemAnimationDetails.positionStrength
                                         : MolecularSystemAnimationDetails.positionStrength *
                                               rnd3(MolecularSystemAnimationDetails.positionSeed + rnd));

    Vector3d startPos;
    Quaterniond startRot;
    const Vector3d startDir = sphereFilling(radius, occurrence, nbOccurrences, rnd, startPos, startRot, offset);

    if (MolecularSystemAnimationDetails.rotationSeed != 0)
        startRot = weightedRandomRotation(startRot, MolecularSystemAnimationDetails.rotationSeed, rnd,
                                          MolecularSystemAnimationDetails.rotationStrength);

    const double endRadius = radius * 2.0;
    const double morphingStep = MolecularSystemAnimationDetails.morphingStep;

    Vector3d endPos = startPos;
    endPos.y = -radius;
    endPos = endPos + (1.0 - (startPos.y + _radius) / endRadius) * Vector3d(endRadius, 0.0, endRadius) *
                          normalize(Vector3d(startDir.x, 0.0, startDir.z));

    Quaterniond endRot{0.0, 0.0, -0.707, 0.707};
    if (MolecularSystemAnimationDetails.rotationSeed != 0)
        endRot = weightedRandomRotation(endRot, MolecularSystemAnimationDetails.rotationSeed, rnd,
                                        MolecularSystemAnimationDetails.rotationStrength);

    const Quaterniond finalRotation = slerp(startRot, endRot, morphingStep);

    const auto finalTranslation = endPos * morphingStep + startPos * (1.0 - morphingStep);
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
    const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails, const double offset) const
{
    Vector3d pos;
    const double diameter = _radius * 2.0;
    do
    {
        pos = Vector3d(rnd1() * diameter, rnd1() * diameter, rnd1() * diameter);
    } while (length(pos) > _radius);

    if (MolecularSystemAnimationDetails.positionSeed != 0)
    {
        const Vector3d posOffset = MolecularSystemAnimationDetails.positionStrength *
                                   Vector3d(rnd2(occurrence + MolecularSystemAnimationDetails.positionSeed),
                                            rnd2(occurrence + MolecularSystemAnimationDetails.positionSeed + 1),
                                            rnd2(occurrence + MolecularSystemAnimationDetails.positionSeed + 2));

        pos += posOffset;
    }
    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    Quaterniond rot = safeQuatlookAt(normalize(pos));
    if (MolecularSystemAnimationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, MolecularSystemAnimationDetails.rotationSeed, occurrence,
                                     MolecularSystemAnimationDetails.rotationStrength);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

} // namespace common
} // namespace bioexplorer
