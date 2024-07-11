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

#include <science/common/Types.h>

namespace bioexplorer
{
namespace meshing
{
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
     * @param model Model into which the 3D representation is created
     * @param pointCloud The list of points
     * @return true If the 3D representation is possible
     * @return false If the 3D representation could not be built
     */
    bool toConvexHull(common::ThreadSafeContainer& container, const PointCloud& pointCloud);
};
} // namespace meshing
} // namespace bioexplorer
