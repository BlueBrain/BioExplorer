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

class SphereShape : public Shape
{
public:
    SphereShape(const Vector4fs& clippingPlanes, const float radius);

    /**
     * @brief getTransformation Provide a random position and rotation on a
     * sphere
     *
     * @param occurence Occurence of the position amongst the maximum of
     * occurences (see next parameters)
     * @return Transformation of the random position and rotation on the sphere
     */
    Transformation getTransformation(const uint64_t occurence,
                                     const uint64_t nbOccurences,
                                     const RandomizationDetails& randDetails,
                                     const float offset) const final;

    /**
     * @brief getTransformation Provide a random position and rotation on a
     * sphere that morphs to a plane
     *
     * @param occurence Occurence of the position amongst the maximum of
     * occurences (see next parameters)
     * @param morphingStep Morphing step between 0 and 1. 0 is sphere, 1 is
     * plane
     * @return Transformation of the random position and rotation on the sphere
     */
    Transformation getTransformation(const uint64_t occurence,
                                     const uint64_t nbOccurences,
                                     const RandomizationDetails& randDetails,
                                     const float offset,
                                     const float morphingStep) const final;

    bool isInside(const Vector3f& point) const final;

private:
    float _radius;
};

} // namespace common
} // namespace bioexplorer
