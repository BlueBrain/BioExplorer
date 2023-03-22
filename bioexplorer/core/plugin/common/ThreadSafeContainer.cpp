/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
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

#include "ThreadSafeContainer.h"

#include "CommonTypes.h"
#include "Utils.h"

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>

#include <iostream>

namespace bioexplorer
{
namespace common
{
using namespace brayns;

ThreadSafeContainer::ThreadSafeContainer(Model& model, const Vector3d& scale)
    : _model(model)
    , _scale(scale)
{
}

uint64_t ThreadSafeContainer::addSphere(
    const Vector3f& position, const float radius, const size_t materialId,
    const bool useSdf, const uint64_t userData, const Neighbours& neighbours,
    const Vector3f displacement)
{
    const Vector3f scale = _scale;
    const Vector3f scaledDisplacement{displacement.x * scale.x,
                                      displacement.y / scale.x, displacement.z};
    if (useSdf)
        return _addSDFGeometry(materialId,
                               createSDFSphere(position * scale,
                                               (radius - displacement.x) *
                                                   scale.x,
                                               userData, scaledDisplacement),
                               neighbours);
    return _addSphere(materialId,
                      {position * scale, radius * scale.x, userData});
}

uint64_t ThreadSafeContainer::addCone(
    const Vector3f& sourcePosition, const float sourceRadius,
    const Vector3f& targetPosition, const float targetRadius,
    const size_t materialId, const bool useSdf, const uint64_t userDataOffset,
    const Neighbours& neighbours, const Vector3f displacement)
{
    const Vector3f scale = _scale;
    const Vector3f scaledDisplacement{displacement.x * scale.x,
                                      displacement.y / scale.x, displacement.z};
    if (useSdf)
    {
        const auto geom =
            createSDFConePill(sourcePosition * scale, targetPosition * scale,
                              (sourceRadius - displacement.x) * scale.x,
                              (targetRadius - displacement.x) * scale.x,
                              userDataOffset, scaledDisplacement);
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

void ThreadSafeContainer::addMesh(const size_t materialId,
                                  const TriangleMesh& mesh)
{
    _meshesMap[materialId] = mesh;
}

void ThreadSafeContainer::addStreamline(const size_t materialId,
                                        const StreamlinesData& streamline)
{
    _streamlinesMap[materialId] = streamline;
}

uint64_t ThreadSafeContainer::_addSphere(const size_t materialId,
                                         const Sphere& sphere)
{
    _spheresMap[materialId].push_back(sphere);
    return 0; // Only used by SDF geometry
}

uint64_t ThreadSafeContainer::_addCylinder(const size_t materialId,
                                           const Cylinder& cylinder)
{
    _cylindersMap[materialId].push_back(cylinder);
    return 0; // Only used by SDF geometry
}

uint64_t ThreadSafeContainer::_addCone(const size_t materialId,
                                       const Cone& cone)
{
    _conesMap[materialId].push_back(cone);
    return 0; // Only used by SDF geometry
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
    _materialIds.clear();
    _commitSpheresToModel();
    _commitCylindersToModel();
    _commitConesToModel();
    _commitSDFGeometriesToModel();
    _commitMeshesToModel();
    _commitStreamlinesToModel();
    _commitMaterials();
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
}

void ThreadSafeContainer::_commitMaterials()
{
    for (const auto materialId : _materialIds)
    {
        Vector3f color{1.f, 1.f, 1.f};
        auto material =
            _model.createMaterial(materialId, std::to_string(materialId));

        material->setDiffuseColor(color);
        material->setSpecularColor(color);
        material->setSpecularExponent(100.f);
        PropertyMap props;
        props.setProperty({MATERIAL_PROPERTY_SHADING_MODE, 0});
        props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});
        props.setProperty({MATERIAL_PROPERTY_CHAMELEON_MODE, 0});
        props.setProperty({MATERIAL_PROPERTY_CAST_USER_DATA, false});
        props.setProperty({MATERIAL_PROPERTY_NODE_ID, 0});
        props.setProperty({MATERIAL_PROPERTY_CLIPPING_MODE, 0});
        material->updateProperties(props);
    }
}

void ThreadSafeContainer::_commitSpheresToModel()
{
    for (const auto& spheres : _spheresMap)
    {
        const auto materialId = spheres.first;
        _materialIds.insert(materialId);
        _model.getSpheres()[materialId].insert(
            _model.getSpheres()[materialId].end(), spheres.second.begin(),
            spheres.second.end());
    }
    _spheresMap.clear();
}

void ThreadSafeContainer::_commitCylindersToModel()
{
    for (const auto& cylinders : _cylindersMap)
    {
        const auto materialId = cylinders.first;
        _materialIds.insert(materialId);
        _model.getCylinders()[materialId].insert(
            _model.getCylinders()[materialId].end(), cylinders.second.begin(),
            cylinders.second.end());
    }
    _cylindersMap.clear();
}

void ThreadSafeContainer::_commitConesToModel()
{
    for (const auto& cones : _conesMap)
    {
        const auto materialId = cones.first;
        _materialIds.insert(materialId);
        _model.getCones()[materialId].insert(
            _model.getCones()[materialId].end(), cones.second.begin(),
            cones.second.end());
    }
    _conesMap.clear();
}

void ThreadSafeContainer::_commitSDFGeometriesToModel()
{
    _finalizeSDFGeometries();

    const uint64_t numGeoms = _sdfMorphologyData.geometries.size();
    size_ts localToGlobalIndex(numGeoms, 0);

    // Add geometries to _model. We do not know the indices of the
    // neighbours yet so we leave them empty.
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

    for (const auto materialId : _sdfMorphologyData.materials)
        _materialIds.insert(materialId);
    _sdfMorphologyData.geometries.clear();
    _sdfMorphologyData.neighbours.clear();
    _sdfMorphologyData.materials.clear();
}

void ThreadSafeContainer::_commitMeshesToModel()
{
    for (const auto& meshes : _meshesMap)
    {
        const auto materialId = meshes.first;
        _materialIds.insert(materialId);
        const auto& srcMesh = meshes.second;
        auto& dstMesh = _model.getTriangleMeshes()[materialId];
        auto vertexOffset = dstMesh.vertices.size();
        dstMesh.vertices.insert(dstMesh.vertices.end(),
                                srcMesh.vertices.begin(),
                                srcMesh.vertices.end());
        auto indexOffset = dstMesh.indices.size();
        dstMesh.indices.insert(dstMesh.indices.end(), srcMesh.indices.begin(),
                               srcMesh.indices.end());
        for (uint64_t i = 0; i < srcMesh.indices.size(); ++i)
            dstMesh.indices[indexOffset + i] += vertexOffset;
        dstMesh.normals.insert(dstMesh.normals.end(), srcMesh.normals.begin(),
                               srcMesh.normals.end());
        dstMesh.colors.insert(dstMesh.colors.end(), srcMesh.colors.begin(),
                              srcMesh.colors.end());
    }
    _meshesMap.clear();
}

void ThreadSafeContainer::_commitStreamlinesToModel()
{
    const auto materialId = 0;
    _materialIds.insert(materialId);
    auto& modelStreamline = _model.getStreamlines()[materialId];
    auto modelOffset = modelStreamline.vertex.size();

    for (const auto& streamline : _streamlinesMap)
    {
        modelStreamline.vertex.insert(modelStreamline.vertex.end(),
                                      streamline.second.vertex.begin(),
                                      streamline.second.vertex.end());
        modelStreamline.vertexColor.insert(
            modelStreamline.vertexColor.end(),
            streamline.second.vertexColor.begin(),
            streamline.second.vertexColor.end());

        for (size_t i = 0; i < streamline.second.vertex.size() - 1; ++i)
            modelStreamline.indices.push_back(modelOffset + i);
        modelOffset += streamline.second.vertex.size();
    }
    _streamlinesMap.clear();
}

} // namespace common
} // namespace bioexplorer
