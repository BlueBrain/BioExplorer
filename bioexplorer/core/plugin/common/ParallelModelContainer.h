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

struct MorphologyInfo
{
    brayns::Vector3d somaPosition;
    brayns::Boxd bounds;
    double maxDistanceToSoma;
};

using MaterialSet = std::set<uint64_t>;
using Neighbours = std::set<size_t>;

const int64_t NO_USER_DATA = -1;

class ParallelModelContainer
{
public:
    ParallelModelContainer(Model& model, const bool useSdf,
                           const Vector3d& scale = Vector3d(1.0, 1.0, 1.0));
    ~ParallelModelContainer() {}

    uint64_t addSphere(const Vector3f& position, const float radius,
                       const size_t materialId, const uint64_t userDataOffset,
                       const Neighbours& neighbours = {},
                       const float displacementRatio = 1.f);

    uint64_t addCone(const Vector3f& sourcePosition, const float sourceRadius,
                     const Vector3f& targetPosition, const float targetRadius,
                     const size_t materialId, const uint64_t userDataOffset,
                     const Neighbours& neighbours = {},
                     const float displacementRatio = 1.f);

    void commitToModel();
    void applyTransformation(const Matrix4f& transformation);

    MorphologyInfo& getMorphologyInfo() { return _morphologyInfo; }

private:
    uint64_t _addSphere(const size_t materialId, const Sphere& sphere);
    uint64_t _addCylinder(const size_t materialId, const Cylinder& cylinder);
    uint64_t _addCone(const size_t materialId, const Cone& cone);
    uint64_t _addSDFGeometry(const size_t materialId, const SDFGeometry& geom,
                             const std::set<size_t>& neighbours);

    void _moveSpheresToModel();
    void _moveCylindersToModel();
    void _moveConesToModel();
    void _moveSDFGeometriesToModel();
    void _createMaterials();
    void _finalizeSDFGeometries();

    SpheresMap _spheres;
    CylindersMap _cylinders;
    ConesMap _cones;
    TriangleMeshMap _trianglesMeshes;
    MorphologyInfo _morphologyInfo;
    SDFMorphologyData _sdfMorphologyData;
    MaterialSet _materialIds;

    Model& _model;
    Vector3d _scale{1.0, 1.0, 1.0};
    bool _useSdf{false};
};
} // namespace common
} // namespace bioexplorer
