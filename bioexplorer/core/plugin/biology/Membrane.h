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
     * @param scene The 3D scene where the glycans are added
     */
    Membrane(Scene &scene);

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
    const ProteinMap &getProteins() const { return _proteins; }

protected:
    Scene &_scene;
    Vector3f _position;
    Quaterniond _rotation;
    ProteinMap _proteins;
};
} // namespace biology
} // namespace bioexplorer
