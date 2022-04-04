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
    MaterialSet materialIds;
    size_t materialId = 0;
    const bool useSdf = true;
    const double displacement = 10.0;

    ThreadSafeContainer modelContainer(model, useSdf);

    auto idx1 =
        modelContainer.addCone(Vector3d(-1, 0, 0), 0.25, Vector3d(0, 0, 0), 0.1,
                               materialId, -1, {}, displacement);
    materialIds.insert(materialId);
    ++materialId;

    auto idx2 =
        modelContainer.addCone(Vector3d(0, 0, 0), 0.1, Vector3d(1, 0, 0), 0.25,
                               materialId, -1, {idx1}, displacement);
    materialIds.insert(materialId);
    ++materialId;

    auto idx3 = modelContainer.addSphere(Vector3d(-1, 0, 0), 0.25, materialId,
                                         -1, {idx1}, displacement);
    materialIds.insert(materialId);
    ++materialId;

    auto idx4 = modelContainer.addSphere(Vector3d(1, 0, 0), 0.25, materialId,
                                         -1, {idx2}, displacement);
    materialIds.insert(materialId);
    ++materialId;

    auto idx5 =
        modelContainer.addCone(Vector3d(0, 0.25, 0), 0.5, Vector3d(0, 1, 0),
                               0.0, materialId, -1, {idx1, idx2}, displacement);
    materialIds.insert(materialId);

    modelContainer.commitToModel();
}
} // namespace common
} // namespace bioexplorer
