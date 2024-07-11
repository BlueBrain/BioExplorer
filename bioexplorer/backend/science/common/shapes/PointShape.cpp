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

#include "PointShape.h"

#include <science/common/Logs.h>
#include <science/common/Utils.h>

namespace bioexplorer
{
namespace common
{
using namespace core;
using namespace details;

PointShape::PointShape(const Vector4ds& clippingPlanes)
    : Shape(clippingPlanes)
{
    _bounds.merge(Vector3d());
}

Transformation PointShape::getTransformation(const uint64_t occurrence, const uint64_t nbOccurrences,
                                             const MolecularSystemAnimationDetails& MolecularSystemAnimationDetails,
                                             const double offset) const
{
    const Vector3d pos{0.f, 0.f, 0.f};

    if (isClipped(pos, _clippingPlanes))
        throw std::runtime_error("Instance is clipped");

    const Quaterniond rot{0, 0, 0, 1};
    Transformation transformation;
    transformation.setTranslation(pos);
    transformation.setRotation(rot);
    return transformation;
}

bool PointShape::isInside(const Vector3d& point) const
{
    PLUGIN_THROW("isInside is not implemented for Plane shapes");
}

} // namespace common
} // namespace bioexplorer
