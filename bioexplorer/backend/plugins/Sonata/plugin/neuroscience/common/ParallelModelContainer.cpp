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

#include "ParallelModelContainer.h"

#include <common/Utils.h>

using namespace core;

namespace sonataexplorer
{
namespace neuroscience
{
namespace common
{
ParallelModelContainer::ParallelModelContainer(const Transformation& transformation)
    : _transformation(transformation)
{
}

void ParallelModelContainer::addSphere(const size_t materialId, const Sphere& sphere)
{
    _spheres[materialId].push_back(sphere);
}

void ParallelModelContainer::addCylinder(const size_t materialId, const Cylinder& cylinder)
{
    _cylinders[materialId].push_back(cylinder);
}

void ParallelModelContainer::addCone(const size_t materialId, const Cone& cone)
{
    _cones[materialId].push_back(cone);
}

void ParallelModelContainer::addSDFGeometry(const size_t materialId, const SDFGeometry& geom,
                                            const std::vector<size_t> neighbours)
{
    _sdfMaterials.push_back(materialId);
    _sdfGeometries.push_back(geom);
    _sdfNeighbours.push_back(neighbours);
}

void ParallelModelContainer::moveGeometryToModel(Model& model)
{
    _moveSpheresToModel(model);
    _moveCylindersToModel(model);
    _moveConesToModel(model);
    _moveSDFGeometriesToModel(model);
    model.mergeBounds(_bounds);
    _sdfMaterials.clear();
}

void ParallelModelContainer::_moveSpheresToModel(Model& model)
{
    for (const auto& sphere : _spheres)
    {
        const auto index = sphere.first;
        model.getSpheres()[index].insert(model.getSpheres()[index].end(), sphere.second.begin(), sphere.second.end());
    }
    _spheres.clear();
}

void ParallelModelContainer::_moveCylindersToModel(Model& model)
{
    for (const auto& cylinder : _cylinders)
    {
        const auto index = cylinder.first;
        model.getCylinders()[index].insert(model.getCylinders()[index].end(), cylinder.second.begin(),
                                           cylinder.second.end());
    }
    _cylinders.clear();
}

void ParallelModelContainer::_moveConesToModel(Model& model)
{
    for (const auto& cone : _cones)
    {
        const auto index = cone.first;
        model.getCones()[index].insert(model.getCones()[index].end(), cone.second.begin(), cone.second.end());
    }
    _cones.clear();
}

void ParallelModelContainer::_moveSDFGeometriesToModel(Model& model)
{
    const size_t numGeoms = _sdfGeometries.size();
    std::vector<size_t> localToGlobalIndex(numGeoms, 0);

    // Add geometries to Model. We do not know the indices of the neighbours
    // yet so we leave them empty.
    for (size_t i = 0; i < numGeoms; i++)
    {
        localToGlobalIndex[i] = model.addSDFGeometry(_sdfMaterials[i], _sdfGeometries[i], {});
    }

    // Write the neighbours using global indices
    uint64_ts neighboursTmp;
    for (uint64_t i = 0; i < numGeoms; i++)
    {
        const uint64_t globalIndex = localToGlobalIndex[i];
        neighboursTmp.clear();

        for (auto localNeighbourIndex : _sdfNeighbours[i])
            neighboursTmp.push_back(localToGlobalIndex[localNeighbourIndex]);

        model.updateSDFGeometryNeighbours(globalIndex, neighboursTmp);
    }
    _sdfGeometries.clear();
    _sdfNeighbours.clear();
}

void ParallelModelContainer::applyTransformation(const PropertyMap& properties, const Matrix4f& transformation)
{
    const auto& translation = _transformation.getTranslation();
    const auto& rotation = _transformation.getRotation();
    for (auto& s : _spheres)
        for (auto& sphere : s.second)
        {
            sphere.center = _getAlignmentToGrid(properties, translation + rotation * transformVector3d(sphere.center,
                                                                                                       transformation));
            _bounds.merge(sphere.center + sphere.radius);
            _bounds.merge(sphere.center - sphere.radius);
        }
    for (auto& c : _cylinders)
        for (auto& cylinder : c.second)
        {
            cylinder.center =
                _getAlignmentToGrid(properties,
                                    translation + rotation * transformVector3d(cylinder.center, transformation));
            cylinder.up = _getAlignmentToGrid(properties,
                                              translation + rotation * transformVector3d(cylinder.up, transformation));
            _bounds.merge(cylinder.center + cylinder.radius);
            _bounds.merge(cylinder.center - cylinder.radius);
            _bounds.merge(cylinder.up + cylinder.radius);
            _bounds.merge(cylinder.up - cylinder.radius);
        }
    for (auto& c : _cones)
        for (auto& cone : c.second)
        {
            cone.center = _getAlignmentToGrid(properties,
                                              translation + rotation * transformVector3d(cone.center, transformation));
            cone.up =
                _getAlignmentToGrid(properties, translation + rotation * transformVector3d(cone.up, transformation));
            _bounds.merge(cone.center + cone.centerRadius);
            _bounds.merge(cone.center - cone.centerRadius);
            _bounds.merge(cone.up + cone.upRadius);
            _bounds.merge(cone.up - cone.upRadius);
        }
    for (auto& s : _sdfGeometries)
    {
        s.p0 = _getAlignmentToGrid(properties, translation + rotation * transformVector3d(s.p0, transformation));
        s.p1 = _getAlignmentToGrid(properties, translation + rotation * transformVector3d(s.p1, transformation));
    }
}

Vector3d ParallelModelContainer::_getAlignmentToGrid(const PropertyMap& properties, const Vector3d& position) const
{
    const double alignToGrid = properties.getProperty<double>(PROP_ALIGN_TO_GRID.name);

    if (alignToGrid <= 0.0)
        return position;

    const Vector3d tmp = Vector3d(Vector3i(position / alignToGrid) * static_cast<int>(alignToGrid));
    return Vector3d(std::floor(tmp.x), std::floor(tmp.y), std::floor(tmp.z));
}

} // namespace common
} // namespace neuroscience
} // namespace sonataexplorer
