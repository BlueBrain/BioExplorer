/*
 * Copyright 2020-2024 Blue Brain Project / EPFL
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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
class ParallelModelContainer
{
public:
    ParallelModelContainer(const core::Transformation& transformation);
    ~ParallelModelContainer() {}

    void addSphere(const size_t materialId, const core::Sphere& sphere);
    void addCylinder(const size_t materialId, const core::Cylinder& cylinder);
    void addCone(const size_t materialId, const core::Cone& cone);
    void addSDFGeometry(const size_t materialId, const core::SDFGeometry& geom, const size_ts neighbours);
    void moveGeometryToModel(core::Model& model);
    void applyTransformation(const core::PropertyMap& properties, const core::Matrix4f& transformation);

    MorphologyInfo& getMorphologyInfo() { return _morphologyInfo; }

private:
    void _moveSpheresToModel(core::Model& model);
    void _moveCylindersToModel(core::Model& model);
    void _moveConesToModel(core::Model& model);
    void _moveSDFGeometriesToModel(core::Model& model);
    core::Vector3d _getAlignmentToGrid(const core::PropertyMap& properties, const core::Vector3d& position) const;

    core::SpheresMap _spheres;
    core::CylindersMap _cylinders;
    core::ConesMap _cones;
    core::TriangleMeshMap _trianglesMeshes;
    MorphologyInfo _morphologyInfo;
    std::vector<core::SDFGeometry> _sdfGeometries;
    std::vector<std::vector<size_t>> _sdfNeighbours;
    size_ts _sdfMaterials;
    core::Transformation _transformation;
    core::Boxd _bounds;
};
} // namespace common
} // namespace neuroscience
} // namespace sonataexplorer
