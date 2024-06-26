/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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