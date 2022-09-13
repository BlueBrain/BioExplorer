/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include "Node.h"
#include "ThreadSafeContainer.h"
#include "Types.h"

#include <brayns/common/geometry/Cone.h>
#include <brayns/common/geometry/Cylinder.h>
#include <brayns/common/geometry/Sphere.h>
#include <brayns/common/geometry/Streamline.h>
#include <brayns/common/geometry/TriangleMesh.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;
using namespace details;

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
    SDFGeometries(const double radiusMultiplier,
                  const Vector3d& scale = Vector3d(1.0, 1.0, 1.0));

    /**
     * @brief Add a simple demo of SDF geometries, mainly for testing purpose
     *
     * @param model Brayns model to which the SDF geometries are added
     */
    void addSDFDemo(Model& model);

protected:
    Vector3d _animatedPosition(const Vector4d& position,
                               const uint64_t index = 0) const;

    CellAnimationDetails _animationDetails;
    double _radiusMultiplier{1.0};
};

} // namespace common
} // namespace bioexplorer