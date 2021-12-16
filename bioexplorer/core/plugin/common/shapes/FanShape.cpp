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

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

FanShape::FanShape(const Vector4fs& clippingPlanes, const float radius)
    : Shape(clippingPlanes)
    , _radius(radius)
{
    const auto r = radius / 2.f;
    _bounds.merge(Vector3f(-r, -r, -r));
    _bounds.merge(Vector3f(r, r, r));
    _surface = 4.f * M_PI * _radius * _radius;
}

Transformation FanShape::getTransformation(
    const uint64_t occurence, const uint64_t nbOccurences,
    const RandomizationDetails& randDetails, const float offset) const
{
    size_t rnd = occurence;
    if (nbOccurences != 0 && randDetails.seed != 0)
        rnd = rand() % nbOccurences;

    // Randomizer
    float R = _radius;
    if (randDetails.seed != 0)
        R *= 1.f + rnd1() / 30.f;

    // Sphere filling
    const float off = 2.f / nbOccurences;
    const float increment = 0.1f * M_PI * (3.f - sqrt(5.f));
    const float y = ((occurence * off) - 1.f) + off / 2.f;
    const float r = sqrt(1.f - pow(y, 2.f));
    const float phi = rnd * increment;
    const float x = cos(phi) * r;
    const float z = sin(phi) * r;
    const Vector3f normal{x, y, z};

    const Vector3f pos = normal * (R + offset);
    const Quaterniond rot = quatLookAt(normal, UP_VECTOR);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

Transformation FanShape::getTransformation(
    const uint64_t occurence, const uint64_t nbOccurences,
    const RandomizationDetails& randDetails, const float offset,
    const float /*morphingStep*/) const
{
    return getTransformation(occurence, nbOccurences, randDetails, offset);
}

bool FanShape::isInside(const Vector3f& point) const
{
    PLUGIN_THROW("isInside is not implemented for Fan shapes");
}

} // namespace common
} // namespace bioexplorer
