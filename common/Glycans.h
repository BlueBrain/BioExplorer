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

#ifndef BIOEXPLORER_GLYCANS_H
#define BIOEXPLORER_GLYCANS_H

#include <api/BioExplorerParams.h>
#include <brayns/engineapi/Model.h>
#include <common/Node.h>
#include <common/types.h>

namespace bioexplorer
{
class Glycans : public Node
{
public:
    Glycans(Scene& scene, const SugarsDescriptor& sd, Vector3fs positions,
            Quaternions rotations);

private:
    void _readAtom(const std::string& line);
    void _buildModel(Model& model);

    SugarsDescriptor _descriptor;
    Vector3fs _positions;
    Quaternions _rotations;
    AtomMap _atomMap;
    Residues _residues;
};
} // namespace bioexplorer
#endif // BIOEXPLORER_GLYCANS_H
