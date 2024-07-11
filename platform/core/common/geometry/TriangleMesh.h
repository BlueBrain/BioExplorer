/*
    Copyright 2015 - 2024 Blue Brain Project / EPFL

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

#include <platform/core/common/Types.h>

namespace core
{
struct TriangleMesh
{
    Vector3fs vertices;
    Vector3fs normals;
    Vector4fs colors;
    std::vector<Vector3ui> indices;
    std::vector<Vector2f> textureCoordinates;

    void clear()
    {
        vertices.clear();
        normals.clear();
        colors.clear();
        indices.clear();
        textureCoordinates.clear();
    }
};

inline TriangleMesh createBox(const Vector3f& minCorner, const Vector3f& maxCorner)
{
    TriangleMesh result;
    const size_t numVertices = 24;
    const size_t numFaces = 12;
    result.vertices.reserve(numVertices);
    result.normals.reserve(numVertices);
    result.indices.reserve(numFaces);
    // Front face
    result.vertices.emplace_back(minCorner.x, maxCorner.y, maxCorner.z);
    result.vertices.emplace_back(maxCorner);
    result.vertices.emplace_back(minCorner.x, minCorner.y, maxCorner.z);
    result.vertices.emplace_back(maxCorner.x, minCorner.y, maxCorner.z);
    result.normals.emplace_back(0.f, 0.f, -1.f);
    result.normals.emplace_back(0.f, 0.f, -1.f);
    result.normals.emplace_back(0.f, 0.f, -1.f);
    result.normals.emplace_back(0.f, 0.f, -1.f);
    result.indices.emplace_back(0, 2, 1);
    result.indices.emplace_back(1, 2, 3);
    // Back face
    result.vertices.emplace_back(minCorner.x, maxCorner.y, minCorner.z);
    result.vertices.emplace_back(maxCorner.x, maxCorner.y, minCorner.z);
    result.vertices.emplace_back(minCorner);
    result.vertices.emplace_back(maxCorner.x, minCorner.y, minCorner.z);
    result.normals.emplace_back(0.f, 0.f, 1.f);
    result.normals.emplace_back(0.f, 0.f, 1.f);
    result.normals.emplace_back(0.f, 0.f, 1.f);
    result.normals.emplace_back(0.f, 0.f, 1.f);
    result.indices.emplace_back(4, 6, 5);
    result.indices.emplace_back(5, 6, 7);
    // Left face
    result.vertices.emplace_back(minCorner.x, maxCorner.y, minCorner.z);
    result.vertices.emplace_back(minCorner.x, maxCorner.y, maxCorner.z);
    result.vertices.emplace_back(minCorner);
    result.vertices.emplace_back(minCorner.x, minCorner.y, maxCorner.z);
    result.normals.emplace_back(-1.f, 0.f, 0.f);
    result.normals.emplace_back(-1.f, 0.f, 0.f);
    result.normals.emplace_back(-1.f, 0.f, 0.f);
    result.normals.emplace_back(-1.f, 0.f, 0.f);
    result.indices.emplace_back(8, 10, 9);
    result.indices.emplace_back(9, 10, 11);
    // Right face
    result.vertices.emplace_back(maxCorner);
    result.vertices.emplace_back(maxCorner.x, maxCorner.y, minCorner.z);
    result.vertices.emplace_back(maxCorner.x, minCorner.y, maxCorner.z);
    result.vertices.emplace_back(maxCorner.x, minCorner.y, minCorner.z);
    result.normals.emplace_back(1.f, 0.f, 0.f);
    result.normals.emplace_back(1.f, 0.f, 0.f);
    result.normals.emplace_back(1.f, 0.f, 0.f);
    result.normals.emplace_back(1.f, 0.f, 0.f);
    result.indices.emplace_back(12, 14, 13);
    result.indices.emplace_back(13, 14, 15);
    // Top face
    result.vertices.emplace_back(minCorner.x, maxCorner.y, minCorner.z);
    result.vertices.emplace_back(maxCorner.x, maxCorner.y, minCorner.z);
    result.vertices.emplace_back(minCorner.x, maxCorner.y, maxCorner.z);
    result.vertices.emplace_back(maxCorner);
    result.normals.emplace_back(0.f, 1.f, 0.f);
    result.normals.emplace_back(0.f, 1.f, 0.f);
    result.normals.emplace_back(0.f, 1.f, 0.f);
    result.normals.emplace_back(0.f, 1.f, 0.f);
    result.indices.emplace_back(16, 18, 17);
    result.indices.emplace_back(17, 18, 19);
    // Bottom face
    result.vertices.emplace_back(maxCorner.x, minCorner.y, minCorner.z);
    result.vertices.emplace_back(minCorner);
    result.vertices.emplace_back(maxCorner.x, minCorner.y, maxCorner.z);
    result.vertices.emplace_back(minCorner.x, minCorner.y, maxCorner.z);
    result.normals.emplace_back(0.f, -1.f, 0.f);
    result.normals.emplace_back(0.f, -1.f, 0.f);
    result.normals.emplace_back(0.f, -1.f, 0.f);
    result.normals.emplace_back(0.f, -1.f, 0.f);
    result.indices.emplace_back(20, 22, 21);
    result.indices.emplace_back(21, 22, 23);

    return result;
}

} // namespace core
