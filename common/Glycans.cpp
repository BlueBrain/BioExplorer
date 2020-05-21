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

#include "Glycans.h"

#include <common/utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
Glycans::Glycans(Scene& scene, const SugarsDescriptor& sd, Vector3fs positions,
                 Quaternions rotations)
    : Node()
    , _descriptor(sd)
    , _positions(positions)
    , _rotations(rotations)
{
    size_t lineIndex{0};

    std::stringstream lines{_descriptor.contents};
    std::string line;

    while (getline(lines, line, '\n'))
        if (line.find("ATOM") == 0 || line.find("HETATM") == 0)
            _readAtom(line);

    auto model = scene.createModel();

    // Build 3d models according to atoms positions (re-centered to origin)
    Boxf bounds;

    // Recenter
    if (_descriptor.recenter)
    {
#if 1
        for (const auto& atom : _atomMap)
            bounds.merge(atom.second.position);
        const auto& center = bounds.getCenter();
        for (auto& atom : _atomMap)
            atom.second.position -= center;
#else
        const auto& center = _atomMap.begin()->second.position;
        for (auto& atom : _atomMap)
            atom.second.position -= center;
#endif
    }

    _buildModel(*model);

    // Metadata
    ModelMetadata metadata;
    metadata["Atoms"] = std::to_string(_atomMap.size());

    const auto& size = bounds.getSize();
    metadata["Size"] = std::to_string(size.x) + ", " + std::to_string(size.y) +
                       ", " + std::to_string(size.z) + " angstroms";
    _modelDescriptor =
        std::make_shared<ModelDescriptor>(std::move(model), _descriptor.name,
                                          _descriptor.contents, metadata);
}

void Glycans::_buildModel(Model& model)
{
    for (size_t i = 0; i < _positions.size(); ++i)
    {
        const auto& position = _positions[i];
        const auto& rotation = _rotations[i];

        // Atoms
        for (const auto& atom : _atomMap)
        {
            if (i == 0)
            {
                auto material =
                    model.createMaterial(atom.first,
                                         std::to_string(atom.first));

                RGBColor rgb{255, 255, 255};
                const auto it = atomColorMap.find(atom.second.element);
                if (it != atomColorMap.end())
                    rgb = (*it).second;

                material->setDiffuseColor(
                    {rgb.r / 255.f, rgb.g / 255.f, rgb.b / 255.f});
            }

            const Vector3f rotatedPosition =
                atom.second.position * glm::toMat3(rotation);

            model.addSphere(atom.first, {position + rotatedPosition,
                                         _descriptor.atomRadiusMultiplier *
                                             atom.second.radius});
        }

        // Sticks
        if (_descriptor.addSticks)
            for (const auto& atom1 : _atomMap)
                for (const auto& atom2 : _atomMap)
                    if (atom1.first != atom2.first)
                    {
                        const auto stick =
                            atom2.second.position - atom1.second.position;

                        if (length(stick) < DEFAULT_STICK_DISTANCE)
                        {
                            const auto center = (atom2.second.position +
                                                 atom1.second.position) /
                                                2.f;
                            const Vector3f rotatedSrc =
                                atom1.second.position * glm::toMat3(rotation);

                            const Vector3f rotatedDst =
                                center * glm::toMat3(rotation);

                            model.addCylinder(
                                atom1.first,
                                {position + rotatedSrc, position + rotatedDst,
                                 _descriptor.atomRadiusMultiplier *
                                     BOND_RADIUS});
                        }
                    }
    }
}

void Glycans::_readAtom(const std::string& line)
{
    // --------------------------------------------------------------------
    // COLUMNS DATA TYPE    FIELD     DEFINITION
    // --------------------------------------------------------------------
    // 1 - 6   Record name  "ATOM "
    // 7 - 11  Integer      serial     Atom serial number
    // 13 - 16 Atom         name       Atom name
    // 17      Character    altLoc     Alternate location indicator
    // 18 - 20 Residue name resName    Residue name
    // 22      Character    chainID    Chain identifier
    // 23 - 26 Integer      resSeq     Residue sequence number
    // 27      AChar        iCode      Code for insertion of residues
    // 31 - 38 Real(8.3)    x          Orthogonal coords for X in angstroms
    // 39 - 46 Real(8.3)    y          Orthogonal coords for Y in Angstroms
    // 47 - 54 Real(8.3)    z          Orthogonal coords for Z in Angstroms
    // 55 - 60 Real(6.2)    occupancy  Occupancy
    // 61 - 66 Real(6.2)    tempFactor Temperature factor
    // 77 - 78 LString(2)   element    Element symbol, right-justified
    // 79 - 80 LString(2)   charge     Charge on the atom
    // --------------------------------------------------------------------

    std::string s = line.substr(21, 1);
    std::string chainId = trim(s);

    const size_t serial = static_cast<size_t>(atoi(line.substr(6, 5).c_str()));

    Atom atom;
    atom.chainId = chainId;

    s = line.substr(12, 4);
    atom.name = trim(s);

    s = line.substr(16, 1);
    atom.altLoc = trim(s);

    s = line.substr(17, 3);
    atom.resName = trim(s);

    _residues.insert(atom.resName);

    atom.reqSeq = static_cast<size_t>(atoi(line.substr(22, 4).c_str()));

    atom.iCode = line.substr(26, 1);

    atom.position.x = static_cast<float>(atof(line.substr(30, 8).c_str()));
    atom.position.y = static_cast<float>(atof(line.substr(38, 8).c_str()));
    atom.position.z = static_cast<float>(atof(line.substr(46, 8).c_str()));

    atom.occupancy = static_cast<float>(atof(line.substr(54, 6).c_str()));

    if (line.length() >= 60)
        atom.tempFactor = static_cast<float>(atof(line.substr(60, 6).c_str()));

    if (line.length() >= 76)
    {
        s = line.substr(76, 2);
        atom.element = trim(s);
    }

    if (line.length() >= 78)
    {
        s = line.substr(78, 2);
        atom.charge = trim(s);
    }

    // Convert position from nanometers
    atom.position = 0.01f * atom.position;

    // Convert radius from angstrom
    atom.radius = DEFAULT_ATOM_RADIUS;
    auto it = atomicRadii.find(atom.element);
    if (it != atomicRadii.end())
        atom.radius = 0.0001f * (*it).second;
    else
    {
        it = atomicRadii.find(atom.name);
        if (it != atomicRadii.end())
            atom.radius = 0.0001f * (*it).second;
    }

    _atomMap.insert(std::make_pair(serial, atom));
}
} // namespace bioexplorer
