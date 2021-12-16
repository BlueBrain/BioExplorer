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

#include "CubeShape.h"

#include <plugin/common/Logs.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

CubeShape::CubeShape(const Vector4fs& clippingPlanes, const Vector3f& size)
    : Shape(clippingPlanes)
    , _size(size)
{
    _bounds.merge(Vector3f(-size.x / 2.f, -size.y / 2.f, -size.z / 2.f));
    _bounds.merge(Vector3f(size.x / 2.f, size.y / 2.f, size.z / 2.f));
    _surface = 2.f * (size.x * size.y) + 2.f * (size.x * size.z) +
               2.f * (size.y * size.z);
}

Transformation CubeShape::getTransformation(
    const uint64_t occurence, const uint64_t nbOccurences,
    const RandomizationDetails& randDetails, const float /*offset*/) const
{
    Vector3f pos =
        Vector3f(rnd1() * _size.x, rnd1() * _size.y, rnd1() * _size.z);
    Quaterniond dir;

    if (randDetails.positionSeed != 0)
    {
        const Vector3f posOffset = randDetails.positionStrength *
                                   Vector3f(rnd2(randDetails.positionSeed),
                                            rnd2(randDetails.positionSeed + 1),
                                            rnd2(randDetails.positionSeed + 2));

        pos += posOffset;
    }

    if (randDetails.rotationSeed != 0)
        dir = randomQuaternion(randDetails.rotationSeed);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(dir);
    return transformation;
}

Transformation CubeShape::getTransformation(
    const uint64_t occurence, const uint64_t nbOccurences,
    const RandomizationDetails& randDetails, const float offset,
    const float /*morphingStep*/) const
{
    return getTransformation(occurence, nbOccurences, randDetails, offset);
}

bool CubeShape::isInside(const Vector3f& point) const
{
    PLUGIN_THROW("isInside is not implemented for Cube shapes");
}

} // namespace common
} // namespace bioexplorer
