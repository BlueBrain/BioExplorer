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

#include <plugin/molecularsystems/Node.h>

namespace bioexplorer
{
namespace molecularsystems
{
using namespace details;
using namespace common;

/**
 * @brief The RNASequence class
 */
class RNASequence : public Node
{
public:
    /**
     * @brief Construct a new RNASequence object
     *
     * @param scene Scene to which the RNA sequence should be added
     * @param details Details of the RNA sequence
     * @param position Relative position of the RNA sequence in the assembly
     */
    RNASequence(Scene& scene, const RNASequenceDetails& details,
                const Vector4ds& clippingPlanes,
                const Vector3d& assemblyPosition = Vector3d(),
                const Quaterniond& assemblyRotation = Quaterniond());

    /**
     * @brief Get the map of RNA sequences
     *
     * @return The map of RNA sequences
     */
    RNASequenceMap getRNASequences() { return _rnaSequenceMap; }

    ProteinPtr getProtein() const { return _protein; }

private:
    void _buildRNAAsProteinInstances(const Quaterniond& rotation);
    void _buildRNAAsCurve(const Quaterniond& rotation);

    Scene& _scene;
    uint64_t _nbElements;
    RNASequenceDetails _details;
    RNASequenceMap _rnaSequenceMap;
    ProteinPtr _protein{nullptr};
    const Vector3d& _assemblyPosition;
    const Quaterniond& _assemblyRotation;
    RNAShapePtr _shape{nullptr};
};
} // namespace molecularsystems
} // namespace bioexplorer
