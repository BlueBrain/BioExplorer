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

#include "SinusoidShape.h"

#include <plugin/common/Logs.h>
#include <plugin/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

SinusoidShape::SinusoidShape(const Vector4fs& clippingPlanes,
                             const Vector3f& size)
    : Shape(clippingPlanes)
    , _size(size)
{
    _bounds.merge(Vector3f(-size.x / 2.f, -size.y / 2.f, -size.z / 2.f));
    _bounds.merge(Vector3f(size.x / 2.f, size.y / 2.f, size.z / 2.f));
    _surface = size.x * size.z;
}

float SinusoidShape::_sinusoide(const float x, const float z) const
{
    return 0.2f * cos(x) * sin(z) + 0.05f * cos(x * 2.3f) * sin(z * 4.6f);
}

Transformation SinusoidShape::getTransformation(
    const uint64_t occurence, const uint64_t nbOccurences,
    const AnimationDetails& animationDetails, const float offset) const
{
    const float step = 0.01f;
    const float angle = 0.01f;
    float upOffset = 0.f;
    if (animationDetails.positionSeed != 0)
        upOffset = animationDetails.positionStrength *
                   rnd3((animationDetails.positionSeed + occurence) * 10);

    const float x = rnd1() * _size.x;
    const float z = rnd1() * _size.z;
    const float y = upOffset + _size.y * _sinusoide(x * angle, z * angle);

    Vector3f pos = Vector3f(x, y, z);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    const Vector3f v1 =
        Vector3f(x + step,
                 upOffset + _size.y * _sinusoide((x + step) * angle, z * angle),
                 z) -
        pos;
    const Vector3f v2 =
        Vector3f(x,
                 upOffset + _size.y * _sinusoide(x * angle, (z + step) * angle),
                 z + step) -
        pos;

    // Rotation
    const Vector3f normal = cross(normalize(v1), normalize(v2));
    Quaterniond rot = quatLookAt(normal, UP_VECTOR);
    if (animationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, animationDetails.rotationSeed,
                                     occurence,
                                     animationDetails.rotationStrength);

    pos += normal * offset;

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool SinusoidShape::isInside(const Vector3f& point) const
{
    PLUGIN_THROW("isInside is not implemented for Sinusoid shapes");
}

} // namespace common
} // namespace bioexplorer
