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
    Membrane(Scene &scene, const Vector3f &assemblyPosition,
             const Quaterniond &assemblyRotation,
             const Vector4fs &clippingPlanes);

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

    virtual bool isInside(const Vector3f &point) const = 0;

protected:
    Scene &_scene;
    const Vector3f _assemblyPosition;
    const Quaterniond _assemblyRotation;
    ProteinMap _lipids;
    Vector4fs _clippingPlanes;
};
} // namespace biology
} // namespace bioexplorer
