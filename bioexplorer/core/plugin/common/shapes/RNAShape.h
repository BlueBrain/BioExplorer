/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

class RNAShape : public Shape
{
public:
    RNAShape(const Vector4fs& clippingPlanes, const RNAShapeType& shapeType,
             const uint64_t nbElements, const Vector2f& shapeParams,
             const Vector2f& valuesRange, const Vector3f& curveParams);

    /**
     * @brief getTransformation Provide a random position and rotation on a
     * sphere
     *
     * @param occurence Occurence of the position amongst the maximum of
     * occurences (see next parameters)
     * @return Transformation of the random position and rotation on the fan
     */
    Transformation getTransformation(const uint64_t occurence,
                                     const uint64_t nbOccurences,
                                     const RandomizationDetails& randDetails,
                                     const float offset) const final;

    Transformation getTransformation(const uint64_t occurence,
                                     const uint64_t nbOccurences,
                                     const RandomizationDetails& randDetails,
                                     const float offset,
                                     const float morphingStep) const final;

    bool isInside(const Vector3f& point) const final;

private:
    void _getSegment(const float u, const float v, Vector3f& src,
                     Vector3f& dst) const;
    Vector3f _trefoilKnot(float t) const;
    Vector3f _torus(float t) const;
    Vector3f _star(float t) const;
    Vector3f _spring(float t) const;
    Vector3f _heart(float u) const;
    Vector3f _thing(float t) const;
    Vector3f _moebius(float u, float v) const;

    RNAShapeType _shapeType;

    Vector3f _U;
    Vector3f _V;
    float _uStep;
    float _vStep;
    float _du;
    float _dv;

    Vector2f _shapeParams;
    Vector2f _valuesRange;
    Vector3f _curveParams;
};

} // namespace common
} // namespace bioexplorer
