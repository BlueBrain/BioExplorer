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

class SinusoidShape : public Shape
{
public:
    SinusoidShape(const Vector4fs& clippingPlanes, const Vector3f& size);

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
                                     const AnimationDetails& animationDetails,
                                     const float offset) const final;

    bool isInside(const Vector3f& point) const final;

private:
    float _sinusoide(const float x, const float z) const;

    Vector3f _size;
};

} // namespace common
} // namespace bioexplorer
