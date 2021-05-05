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

#include "Membrane.h"

namespace bioexplorer
{
namespace biology
{
/**
 * @brief A Membrane object implements a 3D structure of a given shape, but with
 * a surface composed of instances of one or several proteins
 *
 */
class ParametricMembrane : public Membrane
{
public:
    /**
     * @brief Construct a new Membrane object
     *
     * @param scene The 3D scene where the glycans are added
     * @param details The data structure describing the membrane
     */
    ParametricMembrane(Scene& scene, const Vector3f& assemblyPosition,
                       const Quaterniond& assemblyRotation,
                       const Vector4fs& clippingPlanes,
                       const ParametricMembraneDetails& details);

    /**
     * @brief Destroy the Parametric Membrane object
     *
     */
    ~ParametricMembrane();

private:
    void _processInstances();
    std::string _getElementNameFromId(const size_t id);

    ParametricMembraneDetails _details;
};
} // namespace biology
} // namespace bioexplorer
