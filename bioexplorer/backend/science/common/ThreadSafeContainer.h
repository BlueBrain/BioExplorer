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

#pragma once

#include "Types.h"

#include <platform/core/common/CommonTypes.h>

namespace bioexplorer
{
namespace common
{
/**
 * @brief The ThreadSafeContainer class is used to load large datasets in
 * parallel. Every individual element is loaded in a separate thread and
 * eventualy merged into a single Core model
 *
 */
class ThreadSafeContainer
{
public:
    /**
     * @brief Construct a new Thread Safe Model object
     *
     * @param model Core model
     * for the geometry
     * @param scale Scale applied to individual elements
     */
    ThreadSafeContainer(core::Model& model, const double alignToGrid, const core::Vector3d& position,
                        const core::Quaterniond& rotation, const core::Vector3d& scale = core::Vector3d(1.0, 1.0, 1.0));

    /**
     * @brief Destroy the Thread Safe Model object
     *
     */
    ~ThreadSafeContainer() {}

    /**
     * @brief Add a sphere to the thread safe model
     *
     * @param position Position of the sphere
     * @param radius Radius of the sphere
     * @param materialId Material identifier
     * @param useSdf Defines if signed-distance field technique should be used
     * @param userDataOffset User data to attach to the sphere
     * @param neighbours Neighbours identifiers (For signed-distance field
     * geometry)
     * @param displacement Displacement ratio (For signed-distance field
     * geometry)
     * @return uint64_t Index of the geometry in the model
     */
    uint64_t addSphere(const core::Vector3f& position, const float radius, const size_t materialId, const bool useSdf,
                       const uint64_t userDataOffset = 0, const Neighbours& neighbours = {},
                       const core::Vector3f displacement = core::Vector3f());

    /**
     * @brief Add a cut sphere to the thread safe model
     *
     * @param position Position of the sphere
     * @param radius Radius of the sphere
     * @param cutRadius Radius of cut
     * @param materialId Material identifier
     * @param useSdf Defines if signed-distance field technique should be used
     * @param userDataOffset User data to attach to the sphere
     * @param neighbours Neighbours identifiers (For signed-distance field
     * geometry)
     * @param displacement Displacement ratio (For signed-distance field
     * geometry)
     * @return uint64_t Index of the geometry in the model
     */
    uint64_t addCutSphere(const core::Vector3f& position, const float radius, const float cutRadius,
                          const size_t materialId, const uint64_t userDataOffset = NO_USER_DATA,
                          const Neighbours& neighbours = {}, const core::Vector3f displacement = core::Vector3f());

    /**
     * @brief Add a cone to the thread safe model. If both radii are identical
     * and signed-distance field technique is not used, a cylinder is added
     * instead of a cone
     *
     * @param sourcePosition Base position of the cone
     * @param sourceRadius Base radius of the cone
     * @param targetPosition Top position of the cone
     * @param targetRadius Top radius of the cone
     * @param materialId Material identifier
     * @param useSdf Defines if signed-distance field technique should be used
     * @param userDataOffset User data to attach to the sphere
     * @param neighbours Neighbours identifiers (For signed-distance field
     * geometry)
     * @param displacement Displacement ratio (For signed-distance field
     * geometry)
     * @return uint64_t Index of the geometry in the model
     */
    uint64_t addCone(const core::Vector3f& sourcePosition, const float sourceRadius,
                     const core::Vector3f& targetPosition, const float targetRadius, const size_t materialId,
                     const bool useSdf, const uint64_t userDataOffset = NO_USER_DATA, const Neighbours& neighbours = {},
                     const core::Vector3f displacement = core::Vector3f());

    /**
     * @brief Add a cone of spheres to the thread safe model.
     *
     * @param sourcePosition Base position of the cone
     * @param sourceRadius Base radius of the cone
     * @param targetPosition Top position of the cone
     * @param targetRadius Top radius of the cone
     * @param materialId Material identifier
     * @param userDataOffset User data to attach to the sphere
     * @param constantRadius Use contant sphere radius if true
     */
    void addConeOfSpheres(const core::Vector3f& sourcePosition, const float sourceRadius,
                          const core::Vector3f& targetPosition, const float targetRadius, const size_t materialId,
                          const uint64_t userDataOffset = NO_USER_DATA, const float constantRadius = 0.f);

    /**
     * @brief Add a torus to the thread safe model
     *
     * @param position Position of the torus
     * @param outerRadius Outer radius of the torus
     * @param innerRadius Inner radius of the torus
     * @param materialId Material identifier
     * @param useSdf Defines if signed-distance field technique should be used
     * @param userDataOffset User data to attach to the sphere
     * @param neighbours Neighbours identifiers (For signed-distance field geometry)
     * @param displacement Displacement ratio (For signed-distance field geometry)
     * @return uint64_t Index of the geometry in the model
     */
    uint64_t addTorus(const core::Vector3f& position, const float outerRadius, const float innerRadius,
                      const size_t materialId, const uint64_t userDataOffset = NO_USER_DATA,
                      const Neighbours& neighbours = {}, const core::Vector3f displacement = core::Vector3f());

    /**
     * @brief Add a vesica to the thread safe model
     *
     * @param sourcePosition Base position of the vesica
     * @param targetPosition Top position of the vesica
     * @param radius Radius of the vesica
     * @param materialId Material identifier
     * @param useSdf Defines if signed-distance field technique should be used
     * @param userDataOffset User data to attach to the sphere
     * @param neighbours Neighbours identifiers (For signed-distance field geometry)
     * @param displacement Displacement ratio (For signed-distance field geometry)
     * @return uint64_t Index of the geometry in the model
     */
    uint64_t addVesica(const core::Vector3f& sourcePosition, const core::Vector3f& targetPosition, const float radius,
                       const size_t materialId, const uint64_t userDataOffset = NO_USER_DATA,
                       const Neighbours& neighbours = {}, const core::Vector3f displacement = core::Vector3f());

    /**
     * @brief Add a vesica to the thread safe model
     *
     * @param position Position of the ellipsoid
     * @param radii Radii of the vesica
     * @param materialId Material identifier
     * @param useSdf Defines if signed-distance field technique should be used
     * @param userDataOffset User data to attach to the sphere
     * @param neighbours Neighbours identifiers (For signed-distance field geometry)
     * @param displacement Displacement ratio (For signed-distance field geometry)
     * @return uint64_t Index of the geometry in the model
     */
    uint64_t addEllipsoid(const core::Vector3f& position, const core::Vector3f& radii, const size_t materialId,
                          const uint64_t userDataOffset, const Neighbours& neighbours,
                          const core::Vector3f displacement);

    /**
     * @brief Add a mesh to the thread safe model
     *
     * @param mesh Mesh
     */
    void addMesh(const size_t materialId, const core::TriangleMesh& mesh);

    /**
     * @brief Add a streamline to the thread safe model
     *
     * @param streamline Streamline
     */
    void addStreamline(const size_t materialId, const core::StreamlinesData& streamline);

    /**
     * @brief Commit geometries and materials to the Core model
     *
     */
    void commitToModel();

    void setSDFGeometryNeighbours(const uint64_t geometryIndex, const std::set<size_t>& neighbours);

    MaterialSet& getMaterialIds() { return _materialIds; }

private:
    uint64_t _addSphere(const size_t materialId, const core::Sphere& sphere);
    uint64_t _addCylinder(const size_t materialId, const core::Cylinder& cylinder);
    uint64_t _addCone(const size_t materialId, const core::Cone& cone);
    uint64_t _addSDFGeometry(const size_t materialId, const core::SDFGeometry& geom,
                             const std::set<size_t>& neighbours);

    void _commitSpheresToModel();
    void _commitCylindersToModel();
    void _commitConesToModel();
    void _commitSDFGeometriesToModel();
    void _commitMeshesToModel();
    void _commitStreamlinesToModel();
    void _commitMaterials();
    void _finalizeSDFGeometries();

    core::SpheresMap _spheresMap;
    core::CylindersMap _cylindersMap;
    core::ConesMap _conesMap;
    core::TriangleMeshMap _meshesMap;
    core::StreamlinesDataMap _streamlinesMap;
    core::Boxd _bounds;
    core::Model& _model;
    core::Vector3d _position;
    core::Quaterniond _rotation;
    core::Vector3d _scale{1.0, 1.0, 1.0};

    SDFMorphologyData _sdfMorphologyData;
    MaterialSet _materialIds;

    double _alignToGrid{0.0};
};
} // namespace common
} // namespace bioexplorer
