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

#include "Morphologies.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/UniqueId.h>
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
namespace morphology
{
using namespace common;

Morphologies::Morphologies(const double radiusMultiplier, const Vector3f& scale)
    : SDFGeometries(radiusMultiplier == 0.0 ? 1.0 : radiusMultiplier, scale)
{
}

size_t Morphologies::_getNbMitochondrionSegments() const
{
    return 2 + rand() % 18;
}

void Morphologies::_addSomaInternals(const uint64_t index,
                                     ThreadSafeContainer& container,
                                     const size_t baseMaterialId,
                                     const Vector3d& somaPosition,
                                     const double somaRadius,
                                     const double mitochondriaDensity)
{
    // Constants
    const double nucleusDisplacementRatio = 2.0;
    const double mitochondrionRadiusRatio = 0.025;
    const double mitochondrionDisplacementRatio = 20.0;
    const double mitochondrionRadius =
        somaRadius * mitochondrionRadiusRatio; // 5% of the volume of the soma

    // Nucleus
    const double nucleusRadius =
        somaRadius * 0.7; // 70% of the volume of the soma;

    const double somaInnerRadius = nucleusRadius + mitochondrionRadius;
    const double somaOutterRadius = somaRadius * 0.9;
    const double availableVolumeForMitochondria =
        sphereVolume(somaRadius) * mitochondriaDensity;

    const size_t nucleusMaterialId = baseMaterialId + MATERIAL_OFFSET_NUCLEUS;
    container.addSphere(somaPosition, nucleusRadius, nucleusMaterialId,
                        NO_USER_DATA, {}, nucleusDisplacementRatio);

    // Mitochondria
    if (mitochondriaDensity == 0.0)
        return;

    const size_t mitochondrionMaterialId =
        baseMaterialId + MATERIAL_OFFSET_MITOCHONDRION;
    double mitochondriaVolume = 0.0;

    uint64_t geometryIndex = 0;
    while (mitochondriaVolume < availableVolumeForMitochondria)
    {
        const size_t nbSegments = _getNbMitochondrionSegments();
        const auto pointsInSphere =
            getPointsInSphere(nbSegments, somaInnerRadius / somaRadius);
        double previousRadius = mitochondrionRadius;
        for (size_t i = 0; i < nbSegments; ++i)
        {
            // Mitochondrion geometry
            const double radius = (1.0 + (rand() % 500 / 1000.0)) *
                                  mitochondrionRadius * _radiusMultiplier;
            const auto p2 = somaPosition + somaOutterRadius * pointsInSphere[i];

            Neighbours neighbours;
            if (i != 0)
                neighbours = {geometryIndex};
            geometryIndex =
                container.addSphere(p2, radius, mitochondrionMaterialId,
                                    NO_USER_DATA, neighbours,
                                    mitochondrionDisplacementRatio);

            mitochondriaVolume += sphereVolume(radius);

            if (i > 0)
            {
                const auto p1 =
                    somaPosition + somaOutterRadius * pointsInSphere[i - 1];
                geometryIndex =
                    container.addCone(p1, previousRadius, p2, radius,
                                      mitochondrionMaterialId, NO_USER_DATA,
                                      {geometryIndex},
                                      mitochondrionDisplacementRatio);

                mitochondriaVolume +=
                    coneVolume(length(p2 - p1), previousRadius, radius);
            }
            previousRadius = radius;
        }
    }
}

} // namespace morphology
} // namespace bioexplorer
