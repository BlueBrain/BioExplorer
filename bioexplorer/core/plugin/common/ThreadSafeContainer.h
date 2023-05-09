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

#pragma once

#include "Types.h"

namespace bioexplorer
{
namespace common
{
using namespace brayns;

using MaterialSet = std::set<size_t>;
using Neighbours = std::set<size_t>;

/**
 * @brief The ThreadSafeContainer class is used to load large datasets in
 * parallel. Every individual element is loaded in a separate thread and
 * eventualy merged into a single Brayns model
 *
 */
class ThreadSafeContainer
{
public:
    /**
     * @brief Construct a new Thread Safe Model object
     *
     * @param model Brayns model
     * for the geometry
     * @param scale Scale applied to individual elements
     */
    ThreadSafeContainer(Model& model, const double alignToGrid,
                        const Vector3d& position, const Quaterniond& rotation,
                        const Vector3d& scale = Vector3d(1.0, 1.0, 1.0));

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
     * @param userData User data to attach to the sphere
     * @param neighbours Neigbours identifiers (For signed-distance field
     * geometry)
     * @param displacementRatio Displacement ratio (For signed-distance field
     * geometry)
     * @return uint64_t Index of the geometry in the model
     */
    uint64_t addSphere(const Vector3f& position, const float radius,
                       const size_t materialId, const bool useSdf,
                       const uint64_t userDataOffset = 0,
                       const Neighbours& neighbours = {},
                       const Vector3f displacementRatio = Vector3f());

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
     * @param userData User data to attach to the sphere
     * @param neighbours Neigbours identifiers (For signed-distance field
     * geometry)
     * @param displacementRatio Displacement ratio (For signed-distance field
     * geometry)
     * @return uint64_t Index of the geometry in the model
     */
    uint64_t addCone(const Vector3f& sourcePosition, const float sourceRadius,
                     const Vector3f& targetPosition, const float targetRadius,
                     const size_t materialId, const bool useSdf,
                     const uint64_t userDataOffset = 0,
                     const Neighbours& neighbours = {},
                     const Vector3f displacementRatio = Vector3f());

    /**
     * @brief Add a mesh to the thread safe model
     *
     * @param mesh Mesh
     */
    void addMesh(const size_t materialId, const TriangleMesh& mesh);

    /**
     * @brief Add a streamline to the thread safe model
     *
     * @param streamline Streamline
     */
    void addStreamline(const size_t materialId,
                       const StreamlinesData& streamline);

    /**
     * @brief Commit geometries and materials to the Brayns model
     *
     */
    void commitToModel();

    MaterialSet& getMaterialIds() { return _materialIds; }

private:
    uint64_t _addSphere(const size_t materialId, const Sphere& sphere);
    uint64_t _addCylinder(const size_t materialId, const Cylinder& cylinder);
    uint64_t _addCone(const size_t materialId, const Cone& cone);
    uint64_t _addSDFGeometry(const size_t materialId, const SDFGeometry& geom,
                             const std::set<size_t>& neighbours);

    void _commitSpheresToModel();
    void _commitCylindersToModel();
    void _commitConesToModel();
    void _commitSDFGeometriesToModel();
    void _commitMeshesToModel();
    void _commitStreamlinesToModel();
    void _commitMaterials();
    void _finalizeSDFGeometries();

    SpheresMap _spheresMap;
    CylindersMap _cylindersMap;
    ConesMap _conesMap;
    TriangleMeshMap _meshesMap;
    SDFMorphologyData _sdfMorphologyData;
    StreamlinesDataMap _streamlinesMap;
    MaterialSet _materialIds;
    Boxd _bounds;

    Model& _model;
    Vector3d _position;
    Quaterniond _rotation;
    Vector3d _scale{1.0, 1.0, 1.0};
    double _alignToGrid{0.0};
};
} // namespace common
} // namespace bioexplorer
