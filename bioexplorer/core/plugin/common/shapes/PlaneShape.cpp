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

#include "PlaneShape.h"

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

PlaneShape::PlaneShape(const Vector4fs& clippingPlanes, const Vector2f& size)
    : Shape(clippingPlanes)
    , _size(size)
{
    _bounds.merge(Vector3f(-size.x / 2.f, -size.y / 2.f, 0.f));
    _bounds.merge(Vector3f(size.x / 2.f, size.y / 2.f, 0.f));
    _surface = size.x * size.y;
}

Transformation PlaneShape::getTransformation(
    const uint64_t occurence, const uint64_t nbOccurences,
    const AnimationDetails& animationDetails, const float offset) const
{
    float up = 0.f;
    if (animationDetails.seed != 0)
        up = rnd1() * animationDetails.positionStrength;

    Vector3f pos{rnd1() * _size.x, up, rnd1() * _size.y};
    const Quaterniond rot{0.f, 0.f, 0.707f, 0.707f};

    pos += UP_VECTOR * offset;

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool PlaneShape::isInside(const Vector3f& point) const
{
    PLUGIN_THROW("isInside is not implemented for Plane shapes");
}

} // namespace common
} // namespace bioexplorer
