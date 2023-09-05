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
const size_t MATERIAL_OFFSET_SYNAPSE = 5;
const size_t MATERIAL_OFFSET_MITOCHONDRION = 7;
const size_t MATERIAL_OFFSET_NUCLEUS = 8;
const size_t MATERIAL_OFFSET_MYELIN_SHEATH = 9;
const size_t MATERIAL_OFFSET_END_FOOT = 4;
const size_t MATERIAL_OFFSET_MICRO_DOMAIN = 5;

const int64_t SOMA_AS_PARENT = -1;

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

    void _addSomaInternals(common::ThreadSafeContainer& container, const size_t materialId,
                           const core::Vector3d& somaPosition, const double somaRadius,
                           const double mitochondriaDensity, const bool useSdf, const double radiusMultiplier);

    double _getDistanceToSoma(const SectionMap& sections, const Section& section);

    size_t _getMaterialFromDistanceToSoma(const double maxDistanceToSoma, const double distanceToSoma) const;
};
} // namespace morphology
} // namespace bioexplorer