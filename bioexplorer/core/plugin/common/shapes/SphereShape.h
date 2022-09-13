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

#pragma once

#include "Shape.h"

namespace bioexplorer
{
namespace common
{
using namespace details;
using namespace brayns;

class SphereShape : public Shape
{
public:
    /**
     * @brief Construct a new Sphere Shape object
     *
     * @param filled Filled sphere is true, empty sphere if false
     * @param clippingPlanes Clipping planes to apply to the shape
     * @param radius Radius of the sphere
     */
    SphereShape(const bool emptySphere, const Vector4ds& clippingPlanes,
                const double radius);

    /** @copydoc Shape::getTransformation */
    Transformation getTransformation(
        const uint64_t occurrence, const uint64_t nbOccurrences,
        const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
        const double offset) const final;

    /** @copydoc Shape::isInside */
    bool isInside(const Vector3d& point) const final;

private:
    Transformation _getEmptySphereTransformation(
        const uint64_t occurrence, const uint64_t nbOccurrences,
        const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
        const double offset) const;

    Transformation _getEmptySphereMorphedTransformation(
        const uint64_t occurrence, const uint64_t nbOccurrences,
        const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
        const double offset) const;

    Transformation _getFilledSphereTransformation(
        const uint64_t occurrence, const uint64_t nbOccurrences,
        const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
        const double offset) const;

    bool _filled;
    double _radius;
};

} // namespace common
} // namespace bioexplorer
