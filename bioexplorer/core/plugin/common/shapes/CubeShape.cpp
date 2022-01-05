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
#include <plugin/common/Utils.h>

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
    const AnimationDetails& animationDetails, const float /*offset*/) const
{
    Vector3f pos =
        Vector3f(rnd1() * _size.x, rnd1() * _size.y, rnd1() * _size.z);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    Quaterniond dir;

    if (animationDetails.positionSeed != 0)
    {
        const Vector3f posOffset =
            animationDetails.positionStrength *
            Vector3f(rnd2(occurence + animationDetails.positionSeed),
                     rnd2(occurence + animationDetails.positionSeed + 1),
                     rnd2(occurence + animationDetails.positionSeed + 2));

        pos += posOffset;
    }

    if (animationDetails.rotationSeed != 0)
        dir = randomQuaternion(animationDetails.rotationSeed);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(dir);
    return transformation;
}

bool CubeShape::isInside(const Vector3f& point) const
{
    PLUGIN_THROW("isInside is not implemented for Cube shapes");
}

} // namespace common
} // namespace bioexplorer
