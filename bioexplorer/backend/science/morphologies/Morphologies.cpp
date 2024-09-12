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

#include "Morphologies.h"

#include <science/common/UniqueId.h>
#include <science/common/Utils.h>

#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Model.h>

using namespace core;

namespace bioexplorer
{
using namespace common;
using namespace details;

namespace morphology
{
Morphologies::Morphologies(const double alignToGrid, const Vector3d& position, const Quaterniond& rotation,
                           const Vector3f& scale)
    : SDFGeometries(alignToGrid, position, rotation, scale)
{
}

size_t Morphologies::_getNbMitochondrionSegments() const
{
    return 2 + rand() % 5;
}

double Morphologies::_addSomaAsSpheres(const uint64_t neuronId, const size_t somaMaterialId, const SectionMap& sections,
                                       const Vector3d& somaPosition, const Quaterniond& somaRotation,
                                       const double somaRadius, const uint64_t somaUserData,
                                       const double radiusMultiplier, ThreadSafeContainer& container)
{
    Vector3ds sectionRoots;

    double minRadius = std::numeric_limits<double>::max();
    double maxRadius = std::numeric_limits<double>::min();
    double sectionRootRadius = std::numeric_limits<double>::max();
    Vector3d baryCenter;
    for (const auto& section : sections)
        if (section.second.parentId == SOMA_AS_PARENT)
        {
            const auto& points = section.second.points;
            const uint64_t index = std::min(size_t(0), points.size());
            const Vector3d p = points[index];
            sectionRoots.push_back(p);
            const double l = length(p);
            minRadius = std::min(l, minRadius);
            maxRadius = std::max(l, maxRadius);
            sectionRootRadius = std::min(sectionRootRadius, static_cast<double>(points[index].w * 0.5));
            baryCenter += p;
        }
    baryCenter /= sectionRoots.size();
    // sectionRootRadius = sectionRootRadius / sectionRoots.size() * radiusMultiplier;

    const double radius = (minRadius + maxRadius) / 2.0;
    const double somaSurface = 4.0 * glm::pi<double>() * pow(radius, 2.0);
    const double sphereSurfaceOnSoma = glm::pi<double>() * pow(sectionRootRadius, 2.0);
    const uint64_t nbSpheres = somaSurface / sphereSurfaceOnSoma;

    Vector3ds spheres(nbSpheres);

    const double goldenRatio = (1.0 + std::sqrt(5.0)) / 2.0;
    const double angleIncrement = 2.0 * M_PI * goldenRatio;
    for (uint64_t i = 0; i < nbSpheres; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(nbSpheres - 1);
        const double phi = std::acos(1.0 - 2.0 * t);
        const double theta = angleIncrement * static_cast<double>(i);
        const double r = minRadius;
        spheres[i] = Vector3d(r * sin(phi) * cos(theta), r * sin(phi) * sin(theta), r * cos(phi));
    }

    // Smooth soma according to section root
    for (auto& sphere : spheres)
    {
        const double smoothingFactor = 1.0;
        double r = minRadius * smoothingFactor;
        for (const auto& sr : sectionRoots)
        {
            const auto dir = sr - (baryCenter + sphere);
            const double angle = dot(normalize(sr), normalize(baryCenter + sphere));
            if (angle >= 0.0)
                r += length(dir) * -angle * smoothingFactor;
        }

        const auto src =
            _animatedPosition(Vector4d(somaPosition + somaRotation * (baryCenter + normalize(sphere) * r * 0.5),
                                       sectionRootRadius),
                              neuronId);
        container.addSphere(src, sectionRootRadius, somaMaterialId, false, somaUserData);
    }

    return somaRadius;
}

void Morphologies::_addSomaInternals(ThreadSafeContainer& container, const size_t baseMaterialId,
                                     const Vector3d& somaPosition, const double somaRadius,
                                     const double mitochondriaDensity, const bool useSdf, const double radiusMultiplier)
{
    // Nucleus
    //
    // Reference: Age and sex do not affect the volume, cell numbers, or cell
    // size of the suprachiasmatic nucleus of the rat: An unbiased stereological
    // study (https://doi.org/10.1002/cne.903610404)
    const double nucleusRadius = somaRadius * 0.7; // 70% of the volume of the soma;

    const double somaInnerRadius = nucleusRadius + morphology::mitochondrionRadius;
    const double somaOutterRadius = somaRadius * 0.9;
    const double availableVolumeForMitochondria = sphereVolume(somaRadius) * mitochondriaDensity;

    const size_t nucleusMaterialId = baseMaterialId + MATERIAL_OFFSET_NUCLEUS;
    container.addSphere(
        somaPosition, nucleusRadius, nucleusMaterialId, _spheresRepresentation.enabled ? false : useSdf, NO_USER_DATA,
        {},
        Vector3f(nucleusRadius * _getDisplacementValue(DisplacementElement::morphology_nucleus_strength),
                 nucleusRadius * _getDisplacementValue(DisplacementElement::morphology_nucleus_frequency), 0.f));

    // Mitochondria
    if (mitochondriaDensity == 0.0)
        return;

    const size_t mitochondrionMaterialId = baseMaterialId + MATERIAL_OFFSET_MITOCHONDRION;
    double mitochondriaVolume = 0.0;

    uint64_t geometryIndex = 0;
    while (mitochondriaVolume < availableVolumeForMitochondria)
    {
        const size_t nbSegments = _getNbMitochondrionSegments();
        const auto pointsInSphere = getPointsInSphere(nbSegments, somaInnerRadius / somaRadius);
        double previousRadius = mitochondrionRadius;
        double displacementFrequency = 1.0;
        for (size_t i = 0; i < nbSegments; ++i)
        {
            // Mitochondrion geometry
            const double radius =
                (1.2 + (rand() % 500 / 2000.0)) * _getCorrectedRadius(mitochondrionRadius, radiusMultiplier);
            const auto p2 = somaPosition + somaOutterRadius * pointsInSphere[i];

            Neighbours neighbours;
            if (i != 0)
                neighbours = {geometryIndex};
            else
                displacementFrequency =
                    radius * _getDisplacementValue(DisplacementElement::morphology_mitochondrion_frequency);

            geometryIndex =
                container.addSphere(p2, radius, mitochondrionMaterialId, useSdf, NO_USER_DATA, neighbours,
                                    Vector3f(radius * _getDisplacementValue(
                                                          DisplacementElement::morphology_mitochondrion_strength),
                                             displacementFrequency, 0.f));

            mitochondriaVolume += sphereVolume(radius);

            if (i > 0)
            {
                const auto p1 = somaPosition + somaOutterRadius * pointsInSphere[i - 1];
                if (_spheresRepresentation.enabled)
                    container.addConeOfSpheres(p1, previousRadius, p2, radius, mitochondrionMaterialId, NO_USER_DATA,
                                               _spheresRepresentation.radius);
                else
                    geometryIndex = container.addCone(
                        p1, previousRadius, p2, radius, mitochondrionMaterialId, useSdf, NO_USER_DATA, {geometryIndex},
                        Vector3f(radius * _getDisplacementValue(DisplacementElement::morphology_mitochondrion_strength),
                                 displacementFrequency, 0.f));

                mitochondriaVolume += coneVolume(length(p2 - p1), previousRadius, radius);
            }
            previousRadius = radius;
        }
    }
}

double Morphologies::_getDistanceToSoma(const SectionMap& sections, const Section& section)
{
    double distanceToSoma = 0.0;
    auto s = section;
    while (s.parentId != -1)
    {
        const auto it = sections.find(s.parentId);
        if (it == sections.end())
            break;
        s = it->second;
        distanceToSoma += s.length;
    }
    return distanceToSoma;
}

size_t Morphologies::_getMaterialFromDistanceToSoma(const double maxDistanceToSoma, const double distanceToSoma) const
{
    return static_cast<size_t>(512.0 * (1.0 / maxDistanceToSoma) * distanceToSoma);
}

} // namespace morphology
} // namespace bioexplorer
