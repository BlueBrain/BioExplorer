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

#include "SDFGeometries.h"

#include <science/common/UniqueId.h>
#include <science/common/Utils.h>

#include <platform/core/common/utils/Utils.h>
#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>

using namespace core;

namespace bioexplorer
{
using namespace morphology;

namespace common
{
SDFGeometries::SDFGeometries(const double alignToGrid, const Vector3d& position, const Quaterniond& rotation,
                             const Vector3d& scale)
    : Node(scale)
    , _alignToGrid(alignToGrid)
    , _position(position)
    , _rotation(rotation)
{
}

Vector3d SDFGeometries::_animatedPosition(const Vector4d& position, const uint64_t index) const
{
    if (_animationDetails.seed == 0)
        return Vector3d(position);
    const auto seed = _animationDetails.seed + _animationDetails.offset * index;
    const auto amplitude = _animationDetails.amplitude * position.w;
    const auto frequency = _animationDetails.frequency;
    return Vector3d(position.x + amplitude * rnd3(seed + position.x * frequency),
                    position.y + amplitude * rnd3(seed + position.y * frequency),
                    position.z + amplitude * rnd3(seed + position.z * amplitude));
}

Vector4fs SDFGeometries::_getProcessedSectionPoints(const MorphologyRepresentation& representation,
                                                    const Vector4fs& points)
{
    Vector4fs localPoints;
    const float step = 1.f / static_cast<float>(points.size());
    if (representation == MorphologyRepresentation::bezier)
        for (uint64_t i = 0; i <= points.size(); ++i)
            localPoints.push_back(getBezierPoint(points, static_cast<float>(i) * step));
    else
        localPoints = points;
    return localPoints;
}

double SDFGeometries::_getCorrectedRadius(const double radius, const double radiusMultiplier) const
{
    if (radiusMultiplier < 0.0)
        return -radiusMultiplier;
    return radius * radiusMultiplier;
}

void SDFGeometries::addSDFDemo(Model& model)
{
    const bool useSdf = true;
    const Vector3f displacement{0.05f, 10.f, 0.f};

    ThreadSafeContainer modelContainer(model, 0.f, Vector3d(), Quaterniond());

    for (size_t materialId = 0; materialId < 10; ++materialId)
    {
        const float x = materialId * 3.0f;
        Neighbours neighbours;
        neighbours.insert(modelContainer.addSphere(Vector3f(0.f + x, 0.f, 0.f), 0.5f, materialId, useSdf, NO_USER_DATA,
                                                   neighbours, displacement));
        neighbours.insert(modelContainer.addCone(Vector3f(-1.f + x, 0.f, 0.f), 0.25, Vector3d(0.f + x, 0.f, 0.f), 0.1f,
                                                 materialId, useSdf, NO_USER_DATA, neighbours, displacement));
        neighbours.insert(modelContainer.addCone(Vector3f(0.f + x, 0.f, 0.f), 0.1, Vector3f(1.f + x, 0.f, 0.f), 0.25f,
                                                 materialId, useSdf, NO_USER_DATA, neighbours, displacement));
        neighbours.insert(modelContainer.addSphere(Vector3f(-0.5 + x, 0.f, 0.f), 0.25f, materialId, useSdf,
                                                   NO_USER_DATA, neighbours, displacement));
        neighbours.insert(modelContainer.addSphere(Vector3f(0.5 + x, 0.f, 0.f), 0.25f, materialId, useSdf, NO_USER_DATA,
                                                   neighbours, displacement));
        neighbours.insert(modelContainer.addCone(Vector3f(0.f + x, 0.25, 0.f), 0.5f, Vector3f(0.f + x, 1.f, 0.f), 0.f,
                                                 materialId, useSdf, NO_USER_DATA, neighbours, displacement));
        neighbours.insert(modelContainer.addTorus(Vector3f(0.f + x, 0.f, 0.f), 1.5f, 0.5f, materialId, NO_USER_DATA,
                                                  neighbours, displacement));
        neighbours.insert(modelContainer.addCutSphere(Vector3f(0.f + x, 1.f, 0.f), 1.0f, 0.5f, materialId, NO_USER_DATA,
                                                      neighbours, displacement));
        neighbours.insert(modelContainer.addVesica(Vector3f(0.f + x, -1.f, 0.f), Vector3f(0.f + x, -1.5f, 0.f), 1.0f,
                                                   materialId, NO_USER_DATA, neighbours, displacement));
    }
    modelContainer.commitToModel();
    model.applyDefaultColormap();
}

} // namespace common
} // namespace bioexplorer
