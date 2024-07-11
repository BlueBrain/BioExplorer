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
