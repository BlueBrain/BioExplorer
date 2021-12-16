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

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

SphereShape::SphereShape(const Vector4fs& clippingPlanes, const float radius)
    : Shape(clippingPlanes)
    , _radius(radius)
{
    const auto r = radius / 2.f;
    _bounds.merge(Vector3f(-r, -r, -r));
    _bounds.merge(Vector3f(r, r, r));
    _surface = 4.f * M_PI * _radius * _radius;
}

Transformation SphereShape::getTransformation(
    const uint64_t occurence, const uint64_t nbOccurences,
    const RandomizationDetails& randDetails, const float offset) const
{
    size_t rnd = occurence;
    if (nbOccurences != 0 && randDetails.seed != 0)
        rnd = rand() % nbOccurences;

    // Position randomizer
    float R = _radius + (randDetails.positionSeed == 0
                             ? randDetails.positionStrength
                             : randDetails.positionStrength *
                                   rnd3(randDetails.positionSeed + rnd));

    // Sphere filling
    const float off = 2.f / nbOccurences;
    const float increment = M_PI * (3.f - sqrt(5.f));
    const float y = ((occurence * off) - 1.0) + off / 2.0;
    const float r = sqrt(1.0 - pow(y, 2.0));
    const float phi = rnd * increment;
    const float x = cos(phi) * r;
    const float z = sin(phi) * r;

    const Vector3f d{x, y, z};
    const Vector3f pos = d * (R + offset);

    // Rotation
    Quaterniond rot = quatLookAt(d, UP_VECTOR);
    if (randDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, randDetails.rotationSeed, rnd,
                                     randDetails.rotationStrength);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

Transformation SphereShape::getTransformation(
    const uint64_t occurence, const uint64_t nbOccurences,
    const RandomizationDetails& randDetails, const float offset,
    const float morphingStep) const
{
    size_t rnd = occurence;
    if (nbOccurences != 0 && randDetails.seed != 0)
        rnd = rand() % nbOccurences;

    // Position randomizer
    float R = _radius + (randDetails.positionSeed == 0
                             ? randDetails.positionStrength
                             : randDetails.positionStrength *
                                   rnd3(randDetails.positionSeed + rnd));

    // Sphere filling
    const double off = 2.0 / nbOccurences;
    const double increment = M_PI * (3.0 - sqrt(5.0));
    const float y = ((rnd * off) - 1.0) + (off / 2.0);
    const float r = sqrt(1.f - pow(y, 2.0));
    const float phi = rnd * increment;
    const float x = cos(phi) * r;
    const float z = sin(phi) * r;

    const Vector3f startDir{x, y, z};
    const Vector3f startPos = (R + offset) * startDir;
    Vector3f endPos = startPos;

    Quaterniond startRotation = quatLookAt(startDir, UP_VECTOR);
    if (randDetails.rotationSeed != 0)
        startRotation =
            weightedRandomRotation(startRotation, randDetails.rotationSeed, rnd,
                                   randDetails.rotationStrength);

    R = _radius;
    const float endRadius = R * 2.f;

    endPos.y = -R;
    endPos = endPos + (1.f - (startPos.y + (R + offset)) / endRadius) *
                          Vector3f(endRadius, 0.f, endRadius) *
                          normalize(Vector3f(startDir.x, 0.f, startDir.z));

    const Quaterniond endRotation = quatLookAt({0.f, 1.f, 0.f}, UP_VECTOR);
    const Quaterniond finalRotation =
        glm::lerp(startRotation, endRotation, double(morphingStep));

    // Final transformation
    Transformation transformation;
    const Vector3f finalTranslation =
        endPos * morphingStep + startPos * (1.f - morphingStep);
    transformation.setTranslation(finalTranslation);
    transformation.setRotation(finalRotation);
    return transformation;
}

bool SphereShape::isInside(const Vector3f& point) const
{
    return length(point) <= _radius;
}

} // namespace common
} // namespace bioexplorer
