/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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
