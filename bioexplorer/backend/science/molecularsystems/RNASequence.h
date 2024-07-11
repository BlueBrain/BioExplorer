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
