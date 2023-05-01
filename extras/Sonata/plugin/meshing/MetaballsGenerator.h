/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue Brain Project / EPFL
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

#include <brayns/common/types.h>

namespace sonataexplorer
{
namespace meshing
{
using namespace brayns;

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
    void generateMesh(const Vector4fs& metaballs, const size_t gridSize,
                      const float threshold, const size_t defaultMaterialId,
                      TriangleMeshMap& triangles);

private:
    struct SurfaceVertex
    {
        SurfaceVertex()
            : materialId(0)
        {
        }

        Vector3f position;
        Vector3f normal;
        Vector3f texCoords;
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

    void _buildVerticesAndCubes(const Vector4fs& metaballs,
                                const size_t gridSize,
                                const size_t defaultMaterialId,
                                const float scale = 5.f);

    void _buildTriangles(const Vector4fs& metaballs, const float threshold,
                         const size_t defaultMaterialId,
                         TriangleMeshMap& triangles);

    SurfaceVertices _edgeVertices;
    Vertices _vertices;
    Cubes _cubes;
};
} // namespace meshing
} // namespace sonataexplorer
