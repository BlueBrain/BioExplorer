/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2021 Blue BrainProject / EPFL
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

#include <plugin/biology/Node.h>

#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
namespace biology
{
using namespace common;

/**
 * @brief A Membrane object implements a 3D structure of a given shape, but with
 * a surface composed of instances of one or several proteins
 *
 */
class Membrane : public Node
{
public:
    /**
     * @brief Construct a new Membrane object
     *
     * @param scene The 3D scene where the membrane are added
     */
    Membrane(const MembraneDetails &details, Scene &scene,
             const Vector3f &assemblyPosition,
             const Quaterniond &assemblyRotation, const ShapePtr shape,
             const ProteinMap &transmembraneProteins);

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
    void _processInstances();
    std::string _getElementNameFromId(const size_t id) const;

    Scene &_scene;
    MembraneDetails _details;
    uint64_t _nbOccurences;
    const ProteinMap &_transmembraneProteins;
    const Vector3f _assemblyPosition;
    const Quaterniond _assemblyRotation;
    ProteinMap _lipids;
    ShapePtr _shape{nullptr};
};
} // namespace biology
} // namespace bioexplorer
