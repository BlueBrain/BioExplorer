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

#pragma once

#include <plugin/common/SDFGeometries.h>

namespace bioexplorer
{
namespace morphology
{
using namespace brayns;
using namespace common;

const size_t NB_MATERIALS_PER_MORPHOLOGY = 10;
const size_t MATERIAL_OFFSET_SOMA = 1;
const size_t MATERIAL_OFFSET_AXON = 2;
const size_t MATERIAL_OFFSET_DENDRITE = 3;
const size_t MATERIAL_OFFSET_APICAL_DENDRITE = 4;
const size_t MATERIAL_OFFSET_AFFERENT_SYNPASE = 5;
const size_t MATERIAL_OFFSET_EFFERENT_SYNPASE = 6;
const size_t MATERIAL_OFFSET_MITOCHONDRION = 7;
const size_t MATERIAL_OFFSET_NUCLEUS = 8;
const size_t MATERIAL_OFFSET_MYELIN_SHEATH = 9;

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
    Morphologies(const double radiusMultiplier = 1.0,
                 const Vector3f& scale = Vector3d(1.0, 1.0, 1.0));

protected:
    size_t _getNbMitochondrionSegments() const;

    void _addSomaInternals(const uint64_t index, ParallelModelContainer& model,
                           const size_t materialId,
                           const Vector3d& somaPosition,
                           const double somaRadius,
                           const double mitochondriaDensity);
};
} // namespace morphology
} // namespace bioexplorer