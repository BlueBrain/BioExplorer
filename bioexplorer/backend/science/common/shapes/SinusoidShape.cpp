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

#include "SinusoidShape.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace core;
using namespace details;

SinusoidShape::SinusoidShape(const Vector4ds& clippingPlanes, const Vector3d& size)
    : Shape(clippingPlanes)
    , _size(size)
{
    _bounds.merge(Vector3d(-size.x / 2.f, -size.y / 2.f, -size.z / 2.f));
    _bounds.merge(Vector3d(size.x / 2.f, size.y / 2.f, size.z / 2.f));
    _surface = size.x * size.z;
}

double SinusoidShape::_sinusoide(const double x, const double z) const
{
    return 0.2f * cos(x) * sin(z) + 0.05f * cos(x * 2.3f) * sin(z * 4.6f);
}

Transformation SinusoidShape::getTransformation(const uint64_t occurrence, const uint64_t nbOccurrences,
                                                const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
                                                const double offset) const
{
    const double step = 0.01f;
    const double angle = 0.01f;
    double upOffset = 0.f;
    if (MolecularSystemAnimationDetails.positionSeed != 0)
        upOffset = MolecularSystemAnimationDetails.positionStrength *
                   rnd3((MolecularSystemAnimationDetails.positionSeed + occurrence) * 10);

    const double x = rnd1() * _size.x;
    const double z = rnd1() * _size.z;
    const double y = upOffset + _size.y * _sinusoide(x * angle, z * angle);

    Vector3d pos = Vector3d(x, y, z);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    const Vector3d v1 = Vector3d(x + step, upOffset + _size.y * _sinusoide((x + step) * angle, z * angle), z) - pos;
    const Vector3d v2 = Vector3d(x, upOffset + _size.y * _sinusoide(x * angle, (z + step) * angle), z + step) - pos;

    // Rotation
    const Vector3d normal = normalize(cross(normalize(v1), normalize(v2)));
    Quaterniond rot = safeQuatlookAt(normal);
    if (MolecularSystemAnimationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(rot, MolecularSystemAnimationDetails.rotationSeed, occurrence,
                                     MolecularSystemAnimationDetails.rotationStrength);

    pos += normal * offset;

    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool SinusoidShape::isInside(const Vector3d& point) const
{
    PLUGIN_THROW("isInside is not implemented for Sinusoid shapes");
}

} // namespace common
} // namespace bioexplorer
