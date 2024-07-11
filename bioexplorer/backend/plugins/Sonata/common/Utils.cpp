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

#include "Utils.h"

namespace sonataexplorer
{
core::Vector3f get_translation(const core::Matrix4f& matrix)
{
    return core::Vector3f(glm::value_ptr(matrix)[12], glm::value_ptr(matrix)[13], glm::value_ptr(matrix)[14]);
}

bool inBox(const core::Vector3f& point, const core::Boxf& box)
{
    const auto min = box.getMin();
    const auto max = box.getMax();
    return (point.x >= min.x && point.y >= min.y && point.z >= min.z && point.x <= max.x && point.y <= max.y &&
            point.z <= max.z);
}

core::Vector3f getPointInSphere(const float innerRadius)
{
    const float radius = innerRadius + (rand() % 1000 / 1000.f) * (1.f - innerRadius);
    const float phi = M_PI * ((rand() % 2000 - 1000) / 1000.f);
    const float theta = M_PI * ((rand() % 2000 - 1000) / 1000.f);
    core::Vector3f v;
    v.x = radius * sin(phi) * cos(theta);
    v.y = radius * sin(phi) * sin(theta);
    v.z = radius * cos(phi);
    return v;
}

core::Vector3fs getPointsInSphere(const size_t nbPoints, const float innerRadius)
{
    const float radius = innerRadius + (rand() % 1000 / 1000.f) * (1.f - innerRadius);
    float phi = M_PI * ((rand() % 2000 - 1000) / 1000.f);
    float theta = M_PI * ((rand() % 2000 - 1000) / 1000.f);
    core::Vector3fs points;
    for (size_t i = 0; i < nbPoints; ++i)
    {
        core::Vector3f point = {radius * sin(phi) * cos(theta), radius * sin(phi) * sin(theta), radius * cos(phi)};
        points.push_back(point);
        phi += ((rand() % 1000) / 5000.f);
        theta += ((rand() % 1000) / 5000.f);
    }
    return points;
}

core::Vector3d transformVector3d(const core::Vector3f& v, const core::Matrix4f& transformation)
{
    glm::vec3 scale;
    glm::quat rotation;
    glm::vec3 translation;
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(transformation, scale, rotation, translation, skew, perspective);
    return translation + rotation * v;
}

float sphereVolume(const float radius)
{
    return 4.f * M_PI * pow(radius, 3) / 3.f;
}

float cylinderVolume(const float height, const float radius)
{
    return height * M_PI * radius * radius;
}

float coneVolume(const float height, const float r1, const float r2)
{
    return M_PI * (r1 * r1 + r1 * r2 + r2 * r2) * height / 3.f;
}

float capsuleVolume(const float height, const float radius)
{
    return sphereVolume(radius) + cylinderVolume(height, radius);
}

std::vector<uint64_t> GIDsAsInts(const std::string& gids)
{
    std::vector<uint64_t> result;
    std::string split;
    std::istringstream ss(gids);
    while (std::getline(ss, split, ','))
        result.push_back(atoi(split.c_str()));
    return result;
}

} // namespace sonataexplorer
