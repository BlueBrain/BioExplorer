/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include "Shape.h"

namespace bioexplorer
{
namespace common
{
class FanShape : public Shape
{
public:
    /**
     * @brief Construct a new Fan shape object
     *
     * @param clippingPlanes Clipping planes to apply to the shape
     * @param radius Radius of the sphere within which the fan is contained
     */
    FanShape(const Vector4ds& clippingPlanes, const double radius);

    /** @copydoc Shape::getTransformation */
    core::Transformation getTransformation(
        const uint64_t occurrence, const uint64_t nbOccurrences,
        const details::MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
        const double offset) const final;

    /** @copydoc Shape::isInside */
    bool isInside(const core::Vector3d& point) const final;

private:
    double _radius;
};

} // namespace common
} // namespace bioexplorer
