/* Copyright (c) 2020-2021, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: cyrille.favreau@epfl.ch
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <plugin/api/Params.h>
#include <plugin/bioexplorer/Node.h>
#include <plugin/common/Types.h>

#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
using namespace brayns;

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
     * @param descriptor The data structure describing the membrane
     * @param position The position of the membrane in the 3D scene
     * @param clippingPlanes The clipping planes to apply to the membrane
     * @param occupiedDirections The list of directions for which proteins
     * should not be added. This is typically used to remove lipid proteins from
     * areas where other proteins have already been added
     */
    Membrane(Scene &scene, const MembraneDescriptor &descriptor,
             const Vector3f &position, const Vector4fs &clippingPlanes,
             const OccupiedDirections &occupiedDirections);

    /**
     * @brief Destroy the Membrane object
     *
     */
    ~Membrane();

    /**
     * @brief Get the list of proteins defining the membrane
     *
     * @return const ProteinMap& list of proteins defining the membrane
     */
    const ProteinMap &getProteins() const { return _proteins; }

private:
    void _processInstances();
    std::string _getElementNameFromId(const size_t id);

    Scene &_scene;
    Vector3f _position{0.f, 0.f, 0.f};
    MembraneDescriptor _descriptor;
    ProteinMap _proteins;
    Vector4fs _clippingPlanes;
    //    std::map<std::string, std::vector<Transformation>> _transformations;
    const OccupiedDirections &_occupiedDirections;
};
} // namespace bioexplorer
