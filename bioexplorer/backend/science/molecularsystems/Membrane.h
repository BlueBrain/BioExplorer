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
#include <science/common/shapes/Shape.h>

namespace bioexplorer
{
namespace molecularsystems
{
/**
 * @brief A Membrane object implements a 3D structure of a given shape, but with
 * a surface composed of instances of one or several proteins
 *
 */
class Membrane : public common::SDFGeometries
{
public:
    /**
     * @brief Construct a new Membrane object
     *
     * @param scene The 3D scene where the membrane are added
     */
    Membrane(const details::MembraneDetails &details, core::Scene &scene, const core::Vector3d &assemblyPosition,
             const core::Quaterniond &assemblyRotation, const common::ShapePtr shape,
             const molecularsystems::ProteinMap &transmembraneProteins);

    /**
     * @brief Destroy the Membrane object
     *
     */
    virtual ~Membrane();

    /**
     * @brief Get the list of proteins defining the membrane
     *
     * @return const ProteinMap& list of proteins defining the membrane
     */
    const ProteinMap &getLipids() const { return _lipids; }

private:
    double _getDisplacementValue(const DisplacementElement &element) final;

    void _processInstances();
    std::string _getElementNameFromId(const size_t id) const;

    core::Scene &_scene;
    details::MembraneDetails _details;
    uint64_t _nbOccurrences;
    const molecularsystems::ProteinMap &_transmembraneProteins;
    molecularsystems::ProteinMap _lipids;
    common::ShapePtr _shape{nullptr};
};
} // namespace molecularsystems
} // namespace bioexplorer
