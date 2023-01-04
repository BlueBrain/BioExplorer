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

#include <plugin/common/Types.h>

namespace bioexplorer
{
namespace meshing
{
using namespace brayns;

typedef std::map<size_t, Vector4ds> PointCloud;

class PointCloudMesher
{
public:
    /**
     * @brief Construct a new Point Cloud Mesher object
     *
     */
    PointCloudMesher();

    /**
     * @brief Convert a point cloud into a 3D representation using the Convex
     * Hull alogithm
     *
     * @param model Model into which the 3D represenation is created
     * @param pointCloud The list of points
     * @return true If the 3D representation is possible
     * @return false If the 3D representation could not be built
     */
    bool toConvexHull(Model& model, const PointCloud& pointCloud);
};
} // namespace meshing
} // namespace bioexplorer
