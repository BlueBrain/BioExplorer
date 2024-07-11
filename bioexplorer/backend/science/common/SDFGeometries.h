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

#include "Displacement.h"
#include "Node.h"
#include "ThreadSafeContainer.h"
#include "Types.h"

#include <platform/core/common/geometry/Cone.h>
#include <platform/core/common/geometry/Cylinder.h>
#include <platform/core/common/geometry/Sphere.h>
#include <platform/core/common/geometry/Streamline.h>
#include <platform/core/common/geometry/TriangleMesh.h>

namespace bioexplorer
{
namespace common
{
/**
 * @brief The SDFGeometries abstract class is used as a parent to any assembly
 * that potentially requires the signed-distance field technique
 */
class SDFGeometries : public Node
{
public:
    /**
     * @brief Construct a new SDFGeometries object
     *
     */
    SDFGeometries(const double alignToGrid, const core::Vector3d& position = core::Vector3d(0.0, 0.0, 0.0),
                  const core::Quaterniond& rotation = core::Quaterniond(0.0, 0.0, 0.0, 1.0),
                  const core::Vector3d& scale = core::Vector3d(1.0, 1.0, 1.0));

    /**
     * @brief Add a simple demo of SDF geometries, mainly for testing purpose
     *
     * @param model Brayns model to which the SDF geometries are added
     */
    void addSDFDemo(core::Model& model);

protected:
    virtual double _getDisplacementValue(const DisplacementElement& element) { return 0.0; };

    core::Vector4fs _getProcessedSectionPoints(const morphology::MorphologyRepresentation& representation,
                                               const core::Vector4fs& points);

    core::Vector3d _animatedPosition(const core::Vector4d& position, const uint64_t index = 0) const;

    double _getCorrectedRadius(const double radius, const double radiusMultiplier) const;

    details::CellAnimationDetails _animationDetails;
    double _alignToGrid{0.0};
    core::Vector3d _position;
    core::Quaterniond _rotation;
};

} // namespace common
} // namespace bioexplorer