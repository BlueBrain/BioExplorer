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

#pragma once

#include "Shape.h"

namespace bioexplorer
{
namespace common
{
using namespace details;
using namespace brayns;

class PlaneShape : public Shape
{
public:
    /**
     * @brief Construct a new XZ plane shape object
     *
     * @param clippingPlanes Clipping planes to apply to the shape
     * @param size Size of the plane (x and z only)
     */
    PlaneShape(const Vector4ds& clippingPlanes, const Vector2f& size);

    /** @copydoc Shape::getTransformation */
    Transformation getTransformation(
        const uint64_t occurrence, const uint64_t nbOccurrences,
        const MolecularSystemAnimationDetails& animationDetails,
        const double offset) const final;

    /** @copydoc Shape::isInside */
    bool isInside(const Vector3d& point) const final;

private:
    Vector2f _size;
};

} // namespace common
} // namespace bioexplorer
