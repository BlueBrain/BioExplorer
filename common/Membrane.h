/* Copyright (c) 2020, EPFL/Blue Brain Project
 * All rights reserved. Do not distribute without permission.
 * Responsible Author: Cyrille Favreau <cyrille.favreau@epfl.ch>
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

#ifndef BIOEXPLORER_MEMBRANE_H
#define BIOEXPLORER_MEMBRANE_H

#include <api/BioExplorerParams.h>

#include <common/Node.h>
#include <common/types.h>

#include <brayns/engineapi/Model.h>

namespace bioexplorer
{
using namespace brayns;

class Membrane : public Node
{
public:
    Membrane(Scene &scene, const MembraneDescriptor &descriptor,
             const Vector3f &position, const Vector4fs &clippingPlanes,
             const OccupiedDirections &occupiedDirections);
    ~Membrane();

private:
    void _processInstances();
    std::string _getElementNameFromId(const size_t id);

    Scene &_scene;
    Vector3f _position{0.f, 0.f, 0.f};
    MembraneDescriptor _descriptor;
    ProteinMap _proteins;
    Vector4fs _clippingPlanes;
    std::map<std::string, std::vector<Transformation>> _transformations;
    const OccupiedDirections &_occupiedDirections;
};
} // namespace bioexplorer
#endif // BIOEXPLORER_MEMBRANE_H
