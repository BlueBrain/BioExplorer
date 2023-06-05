/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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
