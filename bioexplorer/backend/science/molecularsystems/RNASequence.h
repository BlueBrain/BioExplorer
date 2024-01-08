/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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

#include <science/common/Node.h>
#include <science/common/shapes/RNAShape.h>

namespace bioexplorer
{
namespace molecularsystems
{
/**
 * @brief The RNASequence class
 */
class RNASequence : public common::Node
{
public:
    /**
     * @brief Construct a new RNASequence object
     *
     * @param scene Scene to which the RNA sequence should be added
     * @param details Details of the RNA sequence
     * @param position Relative position of the RNA sequence in the assembly
     */
    RNASequence(core::Scene& scene, const details::RNASequenceDetails& details, const Vector4ds& clippingPlanes,
                const core::Vector3d& assemblyPosition = core::Vector3d(),
                const core::Quaterniond& assemblyRotation = core::Quaterniond());

    /**
     * @brief Get the map of RNA sequences
     *
     * @return The map of RNA sequences
     */
    RNASequenceMap getRNASequences() { return _rnaSequenceMap; }

    ProteinPtr getProtein() const { return _protein; }

private:
    void _buildRNAAsProteinInstances(const core::Quaterniond& rotation);
    void _buildRNAAsCurve(const core::Quaterniond& rotation);

    core::Scene& _scene;
    uint64_t _nbElements;
    details::RNASequenceDetails _details;
    RNASequenceMap _rnaSequenceMap;
    ProteinPtr _protein{nullptr};
    const core::Vector3d& _assemblyPosition;
    const core::Quaterniond& _assemblyRotation;
    common::RNAShapePtr _shape{nullptr};
};
} // namespace molecularsystems
} // namespace bioexplorer
