/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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
