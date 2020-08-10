/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#ifndef BIOEXPLORER_POINTCLOUDMESHER_H
#define BIOEXPLORER_POINTCLOUDMESHER_H

#include <brayns/common/types.h>
#include <map>

namespace bioexplorer
{
using namespace brayns;

typedef std::map<size_t, Vector4fs> PointCloud;

class PointCloudMesher
{
public:
    PointCloudMesher();

    bool toConvexHull(Model& model, const PointCloud& pointCloud);

    bool toMetaballs(brayns::Model& model, const PointCloud& pointCloud,
                     const size_t gridSize, const float threshold);
};

#endif // BIOEXPLORER_POINTCLOUDMESHER_H
}
