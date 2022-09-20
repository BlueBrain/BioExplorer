/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include <plugin/common/CommonTypes.h>
#include <plugin/common/UniqueId.h>
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
namespace common
{
SDFGeometries::SDFGeometries(const double radiusMultiplier,
                             const Vector3d& scale)
    : Node(scale)
    , _radiusMultiplier(radiusMultiplier)
{
}

void SDFGeometries::addSDFDemo(Model& model)
{
    size_t materialId = 0;
    const bool useSdf = true;
    const Vector3f displacement{0.1f, 10.f, 0.f};

    ThreadSafeContainer modelContainer(model, useSdf);
    Neighbours neighbours;
    neighbours.insert(modelContainer.addCone(Vector3d(-1, 0, 0), 0.25,
                                             Vector3d(0, 0, 0), 0.1, materialId,
                                             -1, neighbours, displacement));
    neighbours.insert(
        modelContainer.addCone(Vector3d(0, 0, 0), 0.1, Vector3d(1, 0, 0), 0.25,
                               materialId, -1, neighbours, displacement));
    neighbours.insert(modelContainer.addSphere(Vector3d(-0.5, 0, 0), 0.25,
                                               materialId, -1, neighbours,
                                               displacement));
    neighbours.insert(modelContainer.addSphere(Vector3d(0.5, 0, 0), 0.25,
                                               materialId, -1, neighbours,
                                               displacement));
    neighbours.insert(modelContainer.addCone(Vector3d(0, 0.25, 0), 0.5,
                                             Vector3d(0, 1, 0), 0.0, materialId,
                                             -1, neighbours, displacement));

    modelContainer.commitToModel();
}

Vector3d SDFGeometries::_animatedPosition(const Vector4d& position,
                                          const uint64_t index) const
{
    if (_animationDetails.seed == 0)
        return Vector3d(position);
    const auto seed = _animationDetails.seed + _animationDetails.offset * index;
    const auto amplitude = _animationDetails.amplitude * position.w;
    const auto frequency = _animationDetails.frequency;
    return Vector3d(
        position.x + amplitude * rnd3(seed + position.x * frequency),
        position.y + amplitude * rnd3(seed + position.y * frequency),
        position.z + amplitude * rnd3(seed + position.z * amplitude));
}

} // namespace common
} // namespace bioexplorer
