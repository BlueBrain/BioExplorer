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

#include "HelixShape.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace core;
using namespace details;

HelixShape::HelixShape(const Vector4ds& clippingPlanes, const double radius, const double height)
    : Shape(clippingPlanes)
    , _height(height)
    , _radius(radius)
{
    _bounds.merge(Vector3d(0.0, 0.0, height) + Vector3d(radius, radius, radius));
    _bounds.merge(Vector3d(-radius, -radius, -radius));
    _surface = 2.0 * M_PI * _radius * _radius + _height * (2.0 * M_PI * _radius);
}

Transformation HelixShape::getTransformation(const uint64_t occurrence, const uint64_t nbOccurrences,
                                             const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
                                             const double offset) const
{
    const double radius = _radius + (MolecularSystemAnimationDetails.positionSeed == 0
                                         ? MolecularSystemAnimationDetails.positionStrength
                                         : MolecularSystemAnimationDetails.positionStrength *
                                               rnd3(MolecularSystemAnimationDetails.positionSeed + occurrence));

    const Vector3d pos =
        Vector3d(radius * cos(occurrence), radius * sin(occurrence), -_height * occurrence / nbOccurrences);
    const Vector3d normal = normalize(Vector3d(pos.x, pos.y, 0.0));
    Quaterniond rot = safeQuatlookAt(normal);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    if (MolecularSystemAnimationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, MolecularSystemAnimationDetails.rotationSeed, occurrence,
                                     MolecularSystemAnimationDetails.rotationStrength);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool HelixShape::isInside(const Vector3d& point) const
{
    return length(point) <= _radius;
}

} // namespace common
} // namespace bioexplorer
