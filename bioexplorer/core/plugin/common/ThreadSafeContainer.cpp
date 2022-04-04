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

#include "ThreadSafeContainer.h"

#include "CommonTypes.h"
#include "Utils.h"

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;

ThreadSafeContainer::ThreadSafeContainer(Model& model, const bool useSdf,
                                         const Vector3d& scale)
    : _model(model)
    , _useSdf(useSdf)
    , _scale(scale)
{
}

uint64_t ThreadSafeContainer::addSphere(const Vector3f& position,
                                        const float radius,
                                        const size_t materialId,
                                        const uint64_t userData,
                                        const Neighbours& neighbours,
                                        const float displacementRatio)
{
    _materialIds.insert(materialId);
    const Vector3f scale = _scale;
    if (_useSdf)
    {
        const Vector3f displacementParams = {std::min(radius * scale.x, 0.05f),
                                             displacementRatio / scale.x,
                                             scale.x * 2.0};
        return _addSDFGeometry(materialId,
                               createSDFSphere(position * scale,
                                               radius * scale.x, userData,
                                               displacementParams),
                               neighbours);
    }
    return _addSphere(materialId,
                      {position * scale, radius * scale.x, userData});
}

uint64_t ThreadSafeContainer::addCone(
    const Vector3f& sourcePosition, const float sourceRadius,
    const Vector3f& targetPosition, const float targetRadius,
    const size_t materialId, const uint64_t userDataOffset,
    const Neighbours& neighbours, const float displacementRatio)
{
    _materialIds.insert(materialId);
    const Vector3f scale = _scale;
    if (_useSdf)
    {
        const Vector3f displacementParams = {std::min(sourceRadius * scale.x,
                                                      0.05f),
                                             displacementRatio / scale.x,
                                             scale.x * 2.f};
        const auto geom =
            createSDFConePill(sourcePosition * scale, targetPosition * scale,
                              sourceRadius * scale.x, targetRadius * scale.x,
                              userDataOffset, displacementParams);
        return _addSDFGeometry(materialId, geom, neighbours);
    }
    if (sourceRadius == targetRadius)
        return _addCylinder(materialId,
                            {sourcePosition * scale, targetPosition * scale,
                             sourceRadius * scale.x, userDataOffset});
    return _addCone(materialId, {sourcePosition * scale, targetPosition * scale,
                                 sourceRadius * scale.x, targetRadius * scale.x,
                                 userDataOffset});
}

uint64_t ThreadSafeContainer::_addSphere(const size_t materialId,
                                         const Sphere& sphere)
{
    _spheres[materialId].push_back(sphere);
    return _spheres[materialId].size() - 1;
}

uint64_t ThreadSafeContainer::_addCylinder(const size_t materialId,
                                           const Cylinder& cylinder)
{
    _cylinders[materialId].push_back(cylinder);
    return _cylinders[materialId].size() - 1;
}

uint64_t ThreadSafeContainer::_addCone(const size_t materialId,
                                       const Cone& cone)
{
    _cones[materialId].push_back(cone);
    return _cones[materialId].size() - 1;
}

uint64_t ThreadSafeContainer::_addSDFGeometry(
    const size_t materialId, const SDFGeometry& geom,
    const std::set<size_t>& neighbours)
{
    const uint64_t geometryIndex = _sdfMorphologyData.geometries.size();
    _sdfMorphologyData.geometries.push_back(geom);
    _sdfMorphologyData.neighbours.push_back(neighbours);
    _sdfMorphologyData.materials.push_back(materialId);
    return geometryIndex;
}

void ThreadSafeContainer::commitToModel()
{
    _commitSpheresToModel();
    _commitCylindersToModel();
    _commitConesToModel();
    _commitSDFGeometriesToModel();
    _createMaterials();
}

void ThreadSafeContainer::_finalizeSDFGeometries()
{
    const uint64_t numGeoms = _sdfMorphologyData.geometries.size();
    for (uint64_t i = 0; i < numGeoms; ++i)
    {
        const auto& neighbours = _sdfMorphologyData.neighbours[i];
        for (const auto neighbour : neighbours)
            if (neighbour != i)
                _sdfMorphologyData.neighbours[neighbour].insert(i);
    }

    for (uint64_t i = 0; i < numGeoms; ++i)
    {
        // Convert neighbours from set to vector and erase itself from its
        // neighbours
        size_ts neighbours;
        const auto& neighSet = _sdfMorphologyData.neighbours[i];
        std::copy(neighSet.begin(), neighSet.end(),
                  std::back_inserter(neighbours));
        neighbours.erase(std::remove_if(neighbours.begin(), neighbours.end(),
                                        [i](uint64_t element) {
                                            return element == i;
                                        }),
                         neighbours.end());

        std::set<uint64_t> neighboursSet;
        for (const auto neighbour : neighbours)
            neighboursSet.insert(neighbour);
        _addSDFGeometry(_sdfMorphologyData.materials[i],
                        _sdfMorphologyData.geometries[i], neighboursSet);
    }
}

void ThreadSafeContainer::_createMaterials()
{
    const auto& existingMaterials = _model.getMaterials();
    for (const auto materialId : _materialIds)
    {
        if (existingMaterials.find(materialId) != existingMaterials.end())
            continue;
        Vector3f color{1.f, 1.f, 1.f};
        auto nodeMaterial =
            _model.createMaterial(materialId, std::to_string(materialId));
        nodeMaterial->setDiffuseColor(color);
        nodeMaterial->setSpecularColor(color);
        nodeMaterial->setSpecularExponent(100.f);
        PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE, 0});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        props.setProperty({MATERIAL_PROPERTY_CHAMELEON_MODE, 0});
        props.setProperty({MATERIAL_PROPERTY_CAST_USER_DATA, false});
        props.setProperty({MATERIAL_PROPERTY_NODE_ID, 0});
        nodeMaterial->updateProperties(props);
    }
}

void ThreadSafeContainer::_commitSpheresToModel()
{
    for (const auto& sphere : _spheres)
    {
        const auto index = sphere.first;
        _model.getSpheres()[index].insert(_model.getSpheres()[index].end(),
                                          sphere.second.begin(),
                                          sphere.second.end());
    }
    _spheres.clear();
}

void ThreadSafeContainer::_commitCylindersToModel()
{
    for (const auto& cylinder : _cylinders)
    {
        const auto index = cylinder.first;
        _model.getCylinders()[index].insert(_model.getCylinders()[index].end(),
                                            cylinder.second.begin(),
                                            cylinder.second.end());
    }
    _cylinders.clear();
}

void ThreadSafeContainer::_commitConesToModel()
{
    for (const auto& cone : _cones)
    {
        const auto index = cone.first;
        _model.getCones()[index].insert(_model.getCones()[index].end(),
                                        cone.second.begin(), cone.second.end());
    }
    _cones.clear();
}

void ThreadSafeContainer::_commitSDFGeometriesToModel()
{
    _finalizeSDFGeometries();

    const uint64_t numGeoms = _sdfMorphologyData.geometries.size();
    size_ts localToGlobalIndex(numGeoms, 0);

    // Add geometries to _model. We do not know the indices of the neighbours
    // yet so we leave them empty.
    for (uint64_t i = 0; i < numGeoms; ++i)
        localToGlobalIndex[i] =
            _model.addSDFGeometry(_sdfMorphologyData.materials[i],
                                  _sdfMorphologyData.geometries[i], {});

    // Write the neighbours using global indices
    uint64_ts neighboursTmp;
    for (uint64_t i = 0; i < numGeoms; ++i)
    {
        const uint64_t globalIndex = localToGlobalIndex[i];
        neighboursTmp.clear();

        for (auto localNeighbourIndex : _sdfMorphologyData.neighbours[i])
            neighboursTmp.push_back(localToGlobalIndex[localNeighbourIndex]);

        _model.updateSDFGeometryNeighbours(globalIndex, neighboursTmp);
    }
    _sdfMorphologyData.geometries.clear();
    _sdfMorphologyData.neighbours.clear();
    _sdfMorphologyData.materials.clear();
}

} // namespace common
} // namespace bioexplorer
