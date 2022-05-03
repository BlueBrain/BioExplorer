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

#pragma once

#include "Types.h"

namespace bioexplorer
{
namespace common
{
using namespace brayns;

using MaterialSet = std::set<size_t>;
using Neighbours = std::set<size_t>;

const int64_t NO_USER_DATA = -1;

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
     * @param useSdf Defines if signed-distance field technique should be used
     * for the geometry
     * @param scale Scale applied to individual elements
     */
    ThreadSafeContainer(Model& model, const bool useSdf,
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
     * @param userData User data to attach to the sphere
     * @param neighbours Neigbours identifiers (For signed-distance field
     * geometry)
     * @param displacementRatio Displacement ratio (For signed-distance field
     * geometry)
     * @return uint64_t Index of the geometry in the model
     */
    uint64_t addSphere(const Vector3f& position, const float radius,
                       const size_t materialId,
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
     * @param userData User data to attach to the sphere
     * @param neighbours Neigbours identifiers (For signed-distance field
     * geometry)
     * @param displacementRatio Displacement ratio (For signed-distance field
     * geometry)
     * @return uint64_t Index of the geometry in the model
     */
    uint64_t addCone(const Vector3f& sourcePosition, const float sourceRadius,
                     const Vector3f& targetPosition, const float targetRadius,
                     const size_t materialId, const uint64_t userDataOffset,
                     const Neighbours& neighbours = {},
                     const Vector3f displacementRatio = Vector3f());

    /**
     * @brief Commit geometries and materials to the Brayns model
     *
     */
    void commitToModel();

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
    void _commitMaterials();
    void _finalizeSDFGeometries();

    SpheresMap _spheresMap;
    CylindersMap _cylindersMap;
    ConesMap _conesMap;
    TriangleMeshMap _trianglesMeshesMap;
    SDFMorphologyData _sdfMorphologyData;
    MaterialSet _materialIds;

    Model& _model;
    Vector3d _scale{1.0, 1.0, 1.0};
    bool _useSdf{false};
};
} // namespace common
} // namespace bioexplorer
