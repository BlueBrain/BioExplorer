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

#include "Types.h"

namespace bioexplorer
{
namespace common
{
using MaterialSet = std::set<size_t>;
using Neighbours = std::set<size_t>;

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
     * @brief Add a cone to the thread safe model. If both radii are identical
     * and signed-distance field technique is not used, a cylinder is add
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
                     const bool useSdf, const uint64_t userDataOffset = 0, const Neighbours& neighbours = {},
                     const core::Vector3f displacement = core::Vector3f());

    /**
     * @brief Add a torus to the thread safe model
     *
     * @param sourcePosition Position of the torus
     * @param targetPosition Outer radius of the cone
     * @param targetRadius Inner radius of the cone
     * @param materialId Material identifier
     * @param useSdf Defines if signed-distance field technique should be used
     * @param userDataOffset User data to attach to the sphere
     * @param neighbours Neighbours identifiers (For signed-distance field geometry)
     * @param displacement Displacement ratio (For signed-distance field geometry)
     * @return uint64_t Index of the geometry in the model
     */
    uint64_t addTorus(const core::Vector3f& position, const float outerRadius, const float innerRadius,
                      const size_t materialId, const uint64_t userDataOffset = 0, const Neighbours& neighbours = {},
                      const core::Vector3f displacement = core::Vector3f());
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
