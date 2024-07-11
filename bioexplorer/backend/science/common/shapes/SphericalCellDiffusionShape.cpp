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

#include "SphericalCellDiffusionShape.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace core;
using namespace details;

SphericalCellDiffusionShape::SphericalCellDiffusionShape(const Vector4ds& clippingPlanes, const double radius,
                                                         const double frequency, const double threshold)
    : Shape(clippingPlanes)
    , _radius(radius)
    , _frequency(frequency)
    , _threshold(threshold)
{
    const auto r = radius / 2.0;
    _bounds.merge(Vector3d(-r, -r, -r));
    _bounds.merge(Vector3d(r, r, r));
    _surface = 4.0 * M_PI * _radius * _radius;
}

Transformation SphericalCellDiffusionShape::getTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails, const double offset) const
{
    return _getFilledSphereTransformation(occurrence, nbOccurrences, MolecularSystemAnimationDetails, offset);
}

bool SphericalCellDiffusionShape::isInside(const Vector3d& point) const
{
    return length(point) <= _radius;
}

Transformation SphericalCellDiffusionShape::_getFilledSphereTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails, const double offset) const
{
    Vector3d pos;
    const double diameter = _radius * 2.0;
    do
    {
        pos = Vector3d(rnd1() * diameter, rnd1() * diameter, rnd1() * diameter);
    } while (length(pos) > _radius || worleyNoise(_frequency * pos, 2.f) < _threshold);

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
