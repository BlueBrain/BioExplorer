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

#include <platform/core/common/MathTypes.h>
#include <platform/core/common/Types.h>

#include <glm/gtx/matrix_decompose.hpp>

namespace sonataexplorer
{
// Convertors
core::Vector3f get_translation(const core::Matrix4f& matrix);

core::Vector3d transformVector3d(const core::Vector3f& v, const core::Matrix4f& transformation);

std::vector<uint64_t> GIDsAsInts(const std::string& gids);

// Containers
bool inBox(const core::Vector3f& point, const core::Boxf& box);
core::Vector3f getPointInSphere(const float innerRadius);
core::Vector3fs getPointsInSphere(const size_t nbPoints, const float innerRadius);

// Volumes
float sphereVolume(const float radius);
float cylinderVolume(const float height, const float radius);
float coneVolume(const float height, const float r1, const float r2);
float capsuleVolume(const float height, const float radius);

} // namespace sonataexplorer
