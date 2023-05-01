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

#include "Types.h"

namespace sonataexplorer
{
namespace neuroscience
{
namespace common
{
using namespace brayns;

class ParallelModelContainer
{
public:
    ParallelModelContainer() {}
    ~ParallelModelContainer() {}

    void addSphere(const size_t materialId, const Sphere& sphere);
    void addCylinder(const size_t materialId, const Cylinder& cylinder);
    void addCone(const size_t materialId, const Cone& cone);
    void addSDFGeometry(const size_t materialId, const SDFGeometry& geom,
                        const std::vector<size_t> neighbours);
    void moveGeometryToModel(Model& model);
    void applyTransformation(const PropertyMap& properties,
                             const Matrix4f& transformation);

    MorphologyInfo& getMorphologyInfo() { return _morphologyInfo; }

private:
    void _moveSpheresToModel(Model& model);
    void _moveCylindersToModel(Model& model);
    void _moveConesToModel(Model& model);
    void _moveSDFGeometriesToModel(Model& model);
    Vector3d _getAlignmentToGrid(const PropertyMap& properties,
                                 const Vector3d& position) const;

    SpheresMap _spheres;
    CylindersMap _cylinders;
    ConesMap _cones;
    TriangleMeshMap _trianglesMeshes;
    MorphologyInfo _morphologyInfo;
    std::vector<SDFGeometry> _sdfGeometries;
    std::vector<std::vector<size_t>> _sdfNeighbours;
    std::vector<size_t> _sdfMaterials;
};
} // namespace common
} // namespace neuroscience
} // namespace sonataexplorer
