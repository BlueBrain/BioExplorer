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
