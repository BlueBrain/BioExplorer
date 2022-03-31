/* Copyright (c) 2020-2022, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "ParallelModelContainer.h"

#include "Utils.h"

#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;

size_t ParallelModelContainer::addSphere(const size_t materialId,
                                         const Sphere& sphere)
{
    _spheres[materialId].push_back(sphere);
    return _spheres[materialId].size() - 1;
}

size_t ParallelModelContainer::addCylinder(const size_t materialId,
                                           const Cylinder& cylinder)
{
    _cylinders[materialId].push_back(cylinder);
    return _cylinders[materialId].size() - 1;
}

size_t ParallelModelContainer::addCone(const size_t materialId,
                                       const Cone& cone)
{
    _cones[materialId].push_back(cone);
    return _cones[materialId].size() - 1;
}

void ParallelModelContainer::addSDFGeometry(
    const size_t materialId, const SDFGeometry& geom,
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
    _sdfMaterials.clear();
}

void ParallelModelContainer::_moveSpheresToModel(Model& model)
{
    for (const auto& sphere : _spheres)
    {
        const auto index = sphere.first;
        model.getSpheres()[index].insert(model.getSpheres()[index].end(),
                                         sphere.second.begin(),
                                         sphere.second.end());
    }
    _spheres.clear();
}

void ParallelModelContainer::_moveCylindersToModel(Model& model)
{
    for (const auto& cylinder : _cylinders)
    {
        const auto index = cylinder.first;
        model.getCylinders()[index].insert(model.getCylinders()[index].end(),
                                           cylinder.second.begin(),
                                           cylinder.second.end());
    }
    _cylinders.clear();
}

void ParallelModelContainer::_moveConesToModel(Model& model)
{
    for (const auto& cone : _cones)
    {
        const auto index = cone.first;
        model.getCones()[index].insert(model.getCones()[index].end(),
                                       cone.second.begin(), cone.second.end());
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
        localToGlobalIndex[i] =
            model.addSDFGeometry(_sdfMaterials[i], _sdfGeometries[i], {});

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

void ParallelModelContainer::applyTransformation(const Matrix4f& transformation)
{
    for (auto& s : _spheres)
        for (auto& sphere : s.second)
            sphere.center = transformVector3f(sphere.center, transformation);
    for (auto& c : _cylinders)
        for (auto& cylinder : c.second)
        {
            cylinder.center =
                transformVector3f(cylinder.center, transformation);
            cylinder.up = transformVector3f(cylinder.up, transformation);
        }
    for (auto& c : _cones)
        for (auto& cone : c.second)
        {
            cone.center = transformVector3f(cone.center, transformation);
            cone.up = transformVector3f(cone.up, transformation);
        }
    for (auto& s : _sdfGeometries)
    {
        s.p0 = transformVector3f(s.p0, transformation);
        s.p1 = transformVector3f(s.p1, transformation);
    }
}
} // namespace common
} // namespace bioexplorer
