/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

SinusoidShape::SinusoidShape(const Vector4ds& clippingPlanes,
                             const Vector3d& size)
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

Transformation SinusoidShape::getTransformation(
    const uint64_t occurrence, const uint64_t nbOccurrences,
    const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
    const double offset) const
{
    const double step = 0.01f;
    const double angle = 0.01f;
    double upOffset = 0.f;
    if (MolecularSystemAnimationDetails.positionSeed != 0)
        upOffset =
            MolecularSystemAnimationDetails.positionStrength *
            rnd3((MolecularSystemAnimationDetails.positionSeed + occurrence) *
                 10);

    const double x = rnd1() * _size.x;
    const double z = rnd1() * _size.z;
    const double y = upOffset + _size.y * _sinusoide(x * angle, z * angle);

    Vector3d pos = Vector3d(x, y, z);

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    const Vector3d v1 =
        Vector3d(x + step,
                 upOffset + _size.y * _sinusoide((x + step) * angle, z * angle),
                 z) -
        pos;
    const Vector3d v2 =
        Vector3d(x,
                 upOffset + _size.y * _sinusoide(x * angle, (z + step) * angle),
                 z + step) -
        pos;

    // Rotation
    const Vector3d normal = normalize(cross(normalize(v1), normalize(v2)));
    Quaterniond rot = safeQuatlookAt(normal);
    if (MolecularSystemAnimationDetails.rotationSeed != 0)
        rot = weightedRandomRotation(
            rot, MolecularSystemAnimationDetails.rotationSeed, occurrence,
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
