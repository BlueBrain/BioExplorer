/* Copyright (c) 2020-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: cyrille.favreau@epfl.ch
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <brayns/common/types.h>
#include <map>

namespace bioexplorer
{
using namespace brayns;

typedef std::map<size_t, Vector4fs> PointCloud;

class PointCloudMesher
{
public:
    /**
     * @brief Construct a new Point Cloud Mesher object
     *
     */
    PointCloudMesher();

    /**
     * @brief
     *
     * @param model
     * @param pointCloud
     * @return true
     * @return false
     */
    bool toConvexHull(Model& model, const PointCloud& pointCloud);

    /**
     * @brief
     *
     * @param model
     * @param pointCloud
     * @param gridSize
     * @param threshold
     * @return true
     * @return false
     */
    bool toMetaballs(brayns::Model& model, const PointCloud& pointCloud,
                     const size_t gridSize, const float threshold);
};

} // namespace bioexplorer
