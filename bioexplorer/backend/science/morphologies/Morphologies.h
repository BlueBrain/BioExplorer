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

#pragma once

#include <science/common/SDFGeometries.h>

namespace bioexplorer
{
namespace morphology
{
const double DEFAULT_SPINE_RADIUS = 0.25;

const size_t NB_MATERIALS_PER_MORPHOLOGY = 10;
const size_t MATERIAL_OFFSET_VARICOSITY = 0;
const size_t MATERIAL_OFFSET_SOMA = 1;
const size_t MATERIAL_OFFSET_AXON = 2;
const size_t MATERIAL_OFFSET_DENDRITE = 3;
const size_t MATERIAL_OFFSET_APICAL_DENDRITE = 4;
const size_t MATERIAL_OFFSET_AFFERENT_SYNAPSE = 5;
const size_t MATERIAL_OFFSET_EFFERENT_SYNAPSE = 6;
const size_t MATERIAL_OFFSET_MITOCHONDRION = 7;
const size_t MATERIAL_OFFSET_NUCLEUS = 8;
const size_t MATERIAL_OFFSET_MYELIN_SHEATH = 9;
const size_t MATERIAL_OFFSET_END_FOOT = 4;
const size_t MATERIAL_OFFSET_MICRO_DOMAIN = 5;

/**
 * @brief The Morphologies class
 */
class Morphologies : public common::SDFGeometries
{
public:
    /**
     * @brief Construct a new Morphologies object
     *
     */
    Morphologies(const double alignToGrid, const core::Vector3d& position, const core::Quaterniond& rotation,
                 const core::Vector3f& scale = core::Vector3d(1.0, 1.0, 1.0));

protected:
    size_t _getNbMitochondrionSegments() const;

    double _addSomaAsSpheres(const uint64_t neuronId, const size_t somaMaterialId, const SectionMap& sections,
                             const core::Vector3d& somaPosition, const core::Quaterniond& somaRotation,
                             const double somaRadius, const uint64_t somaUserData, const double radiusMultiplier,
                             common::ThreadSafeContainer& container);

    void _addSomaInternals(common::ThreadSafeContainer& container, const size_t materialId,
                           const core::Vector3d& somaPosition, const double somaRadius,
                           const double mitochondriaDensity, const bool useSdf, const double radiusMultiplier);

    double _getDistanceToSoma(const SectionMap& sections, const Section& section);

    size_t _getMaterialFromDistanceToSoma(const double maxDistanceToSoma, const double distanceToSoma) const;

    common::SpheresRepresentation _spheresRepresentation;
};
} // namespace morphology
} // namespace bioexplorer