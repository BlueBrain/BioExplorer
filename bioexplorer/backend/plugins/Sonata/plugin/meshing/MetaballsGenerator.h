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

#include <platform/core/common/Types.h>

namespace sonataexplorer
{
namespace meshing
{
/**
 * Generated a mesh according to given set of metaballs.
 */
class MetaballsGenerator
{
public:
    MetaballsGenerator() {}
    ~MetaballsGenerator();

    /** Generates a triangle based mesh model according to provided
     * metaballs, grid granularity and threshold
     *
     * @param metaballs metaballs used to generate the mesh
     * @param gridSize Size of the grid
     * @param threshold Points in 3D space that fall below the threshold
     *        (when run through the function) are ONE, while points above the
     *        threshold are ZERO
     * @param defaultMaterialId Default material to apply to the generated mesh
     * @param triangles Generated triangles
     */
    void generateMesh(const core::Vector4fs& metaballs, const size_t gridSize, const float threshold,
                      const size_t defaultMaterialId, core::TriangleMeshMap& triangles);

private:
    struct SurfaceVertex
    {
        SurfaceVertex()
            : materialId(0)
        {
        }

        core::Vector3f position;
        core::Vector3f normal;
        core::Vector3f texCoords;
        size_t materialId;
    };

    struct CubeGridVertex : public SurfaceVertex
    {
        CubeGridVertex()
            : SurfaceVertex()
            , value(0)
        {
        }

        float value; // Value of the scalar field
    };

    struct CubeGridCube
    {
        CubeGridVertex* vertices[8];
    };

    typedef std::vector<CubeGridVertex> Vertices;
    typedef std::vector<CubeGridCube> Cubes;
    typedef std::vector<SurfaceVertex> SurfaceVertices;

    void _clear();

    void _buildVerticesAndCubes(const core::Vector4fs& metaballs, const size_t gridSize, const size_t defaultMaterialId,
                                const float scale = 5.f);

    void _buildTriangles(const core::Vector4fs& metaballs, const float threshold, const size_t defaultMaterialId,
                         core::TriangleMeshMap& triangles);

    SurfaceVertices _edgeVertices;
    Vertices _vertices;
    Cubes _cubes;
};
} // namespace meshing
} // namespace sonataexplorer
