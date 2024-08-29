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
    switch (representation)
    {
    case MorphologyRepresentation::bezier:
    {
        const uint64_t nbBezierPoints = std::max(static_cast<size_t>(3), points.size() / 10);
        const float step = 1.f / static_cast<float>(nbBezierPoints);
        for (float i = 0; i <= 1.f; i += step)
            localPoints.push_back(getBezierPoint(points, i));
        break;
    }
    case MorphologyRepresentation::spheres:
    {
        for (uint64_t i = 0; i < points.size() - 1; ++i)
        {
            const auto p1 = points[i];
            const auto p2 = points[i + 1];
            const auto spheres = fillConeWithSpheres(p1, p2);
            localPoints.insert(localPoints.end(), spheres.begin(), spheres.end());
        }
        break;
    }
    default:
    {
        localPoints = points;
    }
    }
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
    if (rand() % 2 == 0)

        for (size_t materialId = 0; materialId < 10; ++materialId)
        {
            const float x = materialId * 3.0f;
            Neighbours neighbours;
            neighbours.insert(modelContainer.addSphere(Vector3f(0.f + x, 0.f, 0.f), 0.5f, materialId, useSdf,
                                                       NO_USER_DATA, neighbours, displacement));
            neighbours.insert(modelContainer.addCone(Vector3f(-1.f + x, 0.f, 0.f), 0.25, Vector3d(0.f + x, 0.f, 0.f),
                                                     0.1f, materialId, useSdf, NO_USER_DATA, neighbours, displacement));
            neighbours.insert(modelContainer.addCone(Vector3f(0.f + x, 0.f, 0.f), 0.1, Vector3f(1.f + x, 0.f, 0.f),
                                                     0.25f, materialId, useSdf, NO_USER_DATA, neighbours,
                                                     displacement));
            neighbours.insert(modelContainer.addSphere(Vector3f(-0.5 + x, 0.f, 0.f), 0.25f, materialId, useSdf,
                                                       NO_USER_DATA, neighbours, displacement));
            neighbours.insert(modelContainer.addSphere(Vector3f(0.5 + x, 0.f, 0.f), 0.25f, materialId, useSdf,
                                                       NO_USER_DATA, neighbours, displacement));
            neighbours.insert(modelContainer.addCone(Vector3f(0.f + x, 0.25, 0.f), 0.5f, Vector3f(0.f + x, 1.f, 0.f),
                                                     0.f, materialId, useSdf, NO_USER_DATA, neighbours, displacement));
            neighbours.insert(modelContainer.addTorus(Vector3f(0.f + x, 0.f, 0.f), 1.5f, 0.5f, materialId, NO_USER_DATA,
                                                      neighbours, displacement));
            neighbours.insert(modelContainer.addCutSphere(Vector3f(0.f + x, 1.f, 0.f), 1.0f, 0.5f, materialId,
                                                          NO_USER_DATA, neighbours, displacement));
            neighbours.insert(modelContainer.addVesica(Vector3f(0.f + x, -1.f, 0.f), Vector3f(0.f + x, -1.5f, 0.f),
                                                       1.0f, materialId, NO_USER_DATA, neighbours, displacement));
        }
    else
        for (size_t materialId = 0; materialId < 10; ++materialId)
        {
            const float x = materialId * 3.0f;
            Neighbours neighbours;
            neighbours.insert(modelContainer.addSphere(Vector3f(0.f + x, 0.f, 0.f), 1.0f, materialId, useSdf,
                                                       NO_USER_DATA, {}, displacement));
            neighbours.insert(modelContainer.addCone(Vector3f(0.5f + x, 0.f, 0.f), 0.75f, Vector3f(2.f + x, 0.f, 0.f),
                                                     0.f, materialId, useSdf, NO_USER_DATA, {}, displacement));
            neighbours.insert(modelContainer.addCone(Vector3f(-0.5f + x, 0.f, 0.f), 0.75f, Vector3f(-2.f + x, 0.f, 0.f),
                                                     0.f, materialId, useSdf, NO_USER_DATA, {}, displacement));
            neighbours.insert(modelContainer.addCone(Vector3f(0.f + x, 0.5f, 0.f), 0.75f, Vector3f(0.f + x, 2.f, 0.f),
                                                     0.f, materialId, useSdf, NO_USER_DATA, {}, displacement));
            neighbours.insert(modelContainer.addCone(Vector3f(0.f + x, -0.5f, 0.f), 0.75f, Vector3f(0.f + x, -2.f, 0.f),
                                                     0.f, materialId, useSdf, NO_USER_DATA, {}, displacement));

            for (const auto index : neighbours)
                modelContainer.setSDFGeometryNeighbours(index, neighbours);
        }
    modelContainer.commitToModel();
    model.applyDefaultColormap();
}

} // namespace common
} // namespace bioexplorer
