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

#include "PlaneShape.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace core;
using namespace details;

PlaneShape::PlaneShape(const Vector4ds& clippingPlanes, const Vector2f& size)
    : Shape(clippingPlanes)
    , _size(size)
{
    _bounds.merge(Vector3d(-size.x / 2.f, -size.y / 2.f, 0.f));
    _bounds.merge(Vector3d(size.x / 2.f, size.y / 2.f, 0.f));
    _surface = size.x * size.y;
}

Transformation PlaneShape::getTransformation(const uint64_t occurrence, const uint64_t nbOccurrences,
                                             const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
                                             const double offset) const
{
    double up = 0.f;
    if (MolecularSystemAnimationDetails.seed != 0)
        up = rnd1() * MolecularSystemAnimationDetails.positionStrength;

    Vector3d pos{rnd1() * _size.x, up, rnd1() * _size.y};
    const Quaterniond rot{0.0, 0.0, 0.707, 0.707};

    pos += offset * UP_VECTOR;

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool PlaneShape::isInside(const Vector3d& point) const
{
    PLUGIN_THROW("isInside is not implemented for Plane shapes");
}

} // namespace common
} // namespace bioexplorer
