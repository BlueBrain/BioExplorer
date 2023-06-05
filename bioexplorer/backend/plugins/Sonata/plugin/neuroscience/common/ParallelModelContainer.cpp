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

#include "ParallelModelContainer.h"

#include <common/Utils.h>

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
