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
class SurfaceMesher
{
public:
    SurfaceMesher(const uint32_t uuid);

    /** Generates a triangle based mesh model
     *
     * @param points points used to generate the mesh
     * @param triangles Generated triangles
     */
    core::ModelDescriptorPtr generateSurface(core::Scene& scene, const std::string& name, const Vector4ds& points,
                                             const double shrinkfactor = 0.5);

    /** Generates a triangle based mesh model
     *
     * @param points points used to generate the mesh
     * @param triangles Generated triangles
     */
    core::ModelDescriptorPtr generateUnionOfBalls(core::Scene& scene, const std::string& name, const Vector4ds& points);

private:
    uint32_t _uuid;
};
} // namespace meshing
} // namespace bioexplorer
