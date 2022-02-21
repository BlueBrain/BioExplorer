/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include <plugin/common/Types.h>

namespace bioexplorer
{
namespace meshing
{
using namespace brayns;

class SurfaceMesher
{
public:
    SurfaceMesher(const uint32_t uuid);

    /** Generates a triangle based mesh model
     *
     * @param atoms atoms used to generate the mesh
     * @param triangles Generated triangles
     */
    ModelDescriptorPtr generateSurface(brayns::Scene& scene,
                                       const std::string& pdbId,
                                       const Vector4ds& atoms,
                                       const double shrinkfactor = 0.5);

    /** Generates a triangle based mesh model
     *
     * @param atoms atoms used to generate the mesh
     * @param triangles Generated triangles
     */
    ModelDescriptorPtr generateUnionOfBalls(brayns::Scene& scene,
                                            const std::string& pdbId,
                                            const Vector4ds& atoms);

private:
    uint32_t _uuid;
};
} // namespace meshing
} // namespace bioexplorer
