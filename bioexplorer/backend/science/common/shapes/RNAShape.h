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
using namespace core;

class RNAShape : public Shape
{
public:
    /**
     * @brief Construct a new RNAShape object
     *
     * @param clippingPlanes Clipping planes to apply to the shape
     * @param shapeType Type of shape (Trefoil knot, star, spring, etc)
     * @param nbElements Number of elements in the RNA sequence
     * @param shapeParams Size of the shape
     * @param valuesRange Range of values for t
     * @param curveParams Curve parameters based on t, and depending on the
     * shape type
     */
    RNAShape(const Vector4ds& clippingPlanes, const RNAShapeType& shapeType, const uint64_t nbElements,
             const Vector2f& shapeParams, const Vector2f& valuesRange, const Vector3d& curveParams);

    /** @copydoc Shape::getTransformation */
    Transformation getTransformation(const uint64_t occurrence, const uint64_t nbOccurrences,
                                     const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
                                     const double offset) const final;

    /** @copydoc Shape::isInside */
    bool isInside(const Vector3d& point) const final;

private:
    void _getSegment(const double u, const double v, Vector3d& src, Vector3d& dst) const;
    Vector3d _trefoilKnot(double t) const;
    Vector3d _torus(double t) const;
    Vector3d _star(double t) const;
    Vector3d _spring(double t) const;
    Vector3d _heart(double u) const;
    Vector3d _thing(double t) const;
    Vector3d _moebius(double u, double v) const;

    RNAShapeType _shapeType;

    Vector3d _U;
    Vector3d _V;
    double _step;

    Vector2d _shapeParams;
    Vector2d _valuesRange;
    Vector3d _curveParams;
};
typedef std::shared_ptr<RNAShape> RNAShapePtr;

} // namespace common
} // namespace bioexplorer
