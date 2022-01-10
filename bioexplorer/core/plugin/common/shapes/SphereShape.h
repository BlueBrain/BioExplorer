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
    SphereShape(const Vector4ds& clippingPlanes, const double radius);

    /**
     * @brief getTransformation Provide a random position and rotation on a
     * sphere
     *
     * @param occurrence occurrence of the position amongst the maximum of
     * occurrences (see next parameters)
     * @return Transformation of the random position and rotation on the sphere
     */
    Transformation getTransformation(const uint64_t occurrence,
                                     const uint64_t nbOccurrences,
                                     const AnimationDetails& animationDetails,
                                     const double offset) const final;

    bool isInside(const Vector3d& point) const final;

private:
    Transformation _getTransformation(const uint64_t occurrence,
                                      const uint64_t nbOccurrences,
                                      const AnimationDetails& animationDetails,
                                      const double offset) const;

    Transformation _getMorphedTransformation(
        const uint64_t occurrence, const uint64_t nbOccurrences,
        const AnimationDetails& animationDetails, const double offset) const;

    double _radius;
};

} // namespace common
} // namespace bioexplorer
