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

#include "FanShape.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace core;
using namespace details;

FanShape::FanShape(const Vector4ds& clippingPlanes, const double radius)
    : Shape(clippingPlanes)
    , _radius(radius)
{
    const auto r = radius / 2.f;
    _bounds.merge(Vector3d(-r, -r, -r));
    _bounds.merge(Vector3d(r, r, r));
    _surface = 4.f * M_PI * _radius * _radius;
}

Transformation FanShape::getTransformation(const uint64_t occurrence, const uint64_t nbOccurrences,
                                           const MolecularSystemAnimationDetails& molecularSystemAnimationDetails,
                                           const double offset) const
{
    uint64_t rnd = occurrence;
    if (nbOccurrences != 0 && molecularSystemAnimationDetails.seed != 0)
    {
        if (GeneralSettings::getInstance()->getV1Compatibility())
            rnd = rand() % nbOccurrences;
        else
            rnd = rand() % std::numeric_limits<uint64_t>::max();
    }

    const double radius = _radius + (molecularSystemAnimationDetails.positionSeed == 0
                                         ? molecularSystemAnimationDetails.positionStrength
                                         : molecularSystemAnimationDetails.positionStrength *
                                               rnd3(molecularSystemAnimationDetails.positionSeed + rnd));

    Vector3d pos;
    Quaterniond rot;
    sphereFilling(radius + offset, occurrence, nbOccurrences, rnd, pos, rot, 0.1);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    if (molecularSystemAnimationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, molecularSystemAnimationDetails.rotationSeed, rnd,
                                     molecularSystemAnimationDetails.rotationStrength);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool FanShape::isInside(const Vector3d& point) const
{
    PLUGIN_THROW("isInside is not implemented for Fan shapes");
}

} // namespace common
} // namespace bioexplorer
