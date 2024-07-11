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

#include "CubeShape.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace core;
using namespace details;

CubeShape::CubeShape(const Vector4ds& clippingPlanes, const Vector3d& size)
    : Shape(clippingPlanes)
    , _size(size)
{
    _bounds.merge(Vector3d(-size.x / 2.f, -size.y / 2.f, -size.z / 2.f));
    _bounds.merge(Vector3d(size.x / 2.f, size.y / 2.f, size.z / 2.f));
    _surface = 2.f * (size.x * size.y) + 2.f * (size.x * size.z) + 2.f * (size.y * size.z);
}

Transformation CubeShape::getTransformation(const uint64_t occurrence, const uint64_t nbOccurrences,
                                            const MolecularSystemAnimationDetails& molecularSystemAnimationDetails,
                                            const double /*offset*/) const
{
    Vector3d pos = Vector3d(rnd1() * _size.x, rnd1() * _size.y, rnd1() * _size.z);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    Quaterniond dir;

    if (molecularSystemAnimationDetails.positionSeed != 0)
    {
        const Vector3d posOffset = molecularSystemAnimationDetails.positionStrength *
                                   Vector3d(rnd2(occurrence + molecularSystemAnimationDetails.positionSeed),
                                            rnd2(occurrence + molecularSystemAnimationDetails.positionSeed + 1),
                                            rnd2(occurrence + molecularSystemAnimationDetails.positionSeed + 2));

        pos += posOffset;
    }

    if (molecularSystemAnimationDetails.rotationSeed != 0)
        dir = randomQuaternion(occurrence + molecularSystemAnimationDetails.rotationSeed);

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(dir);
    return transformation;
}

bool CubeShape::isInside(const Vector3d& point) const
{
    PLUGIN_THROW("isInside is not implemented for Cube shapes");
}

} // namespace common
} // namespace bioexplorer
