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

#include <map>
#include <platform/core/common/Types.h>

namespace sonataexplorer
{
namespace meshing
{
typedef std::map<size_t, std::vector<core::Vector4f>> PointCloud;

class PointCloudMesher
{
public:
    PointCloudMesher();

    bool toConvexHull(core::Model& model, const PointCloud& pointCloud);

    bool toMetaballs(core::Model& model, const PointCloud& pointCloud, const size_t gridSize, const float threshold);
};
} // namespace meshing
} // namespace sonataexplorer
