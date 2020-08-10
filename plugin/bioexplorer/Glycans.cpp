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

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Scene.h>

namespace bioexplorer
{
Glycans::Glycans(Scene& scene, const SugarsDescriptor& sd, Vector3fs positions,
                 Quaternions rotations)
    : Molecule({})
    , _descriptor(sd)
    , _positions(positions)
    , _rotations(rotations)
{
    size_t lineIndex{0};

    std::stringstream lines{_descriptor.contents};
    std::string line;
    std::string title{sd.name};
    std::string header{sd.name};

    while (getline(lines, line, '\n'))
    {
        if (line.find(KEY_ATOM) == 0 || line.find(KEY_HETATM) == 0)
            _readAtom(line, true);
        else if (line.find(KEY_HEADER) == 0)
            header = _readHeader(line);
        else if (line.find(KEY_TITLE) == 0)
            title = _readTitle(line);
    }
    auto model = scene.createModel();

    // Build 3d models according to atoms positions (re-centered to origin)
    Boxf bounds;

    // Recenter
    if (_descriptor.recenter)
    {
        for (const auto& atom : _atomMap)
            bounds.merge(atom.second.position);
        const auto& center = bounds.getCenter();
        for (auto& atom : _atomMap)
            atom.second.position -= center;
    }

    _buildModel(*model);

    // Metadata
    ModelMetadata metadata;
    metadata[METADATA_ASSEMBLY] = _descriptor.assemblyName;
    metadata[METADATA_TITLE] = title;
    metadata[METADATA_HEADER] = header;
    metadata[METADATA_ATOMS] = std::to_string(_atomMap.size());

    const auto& size = bounds.getSize();
    metadata[METADATA_SIZE] = std::to_string(size.x) + ", " +
                              std::to_string(size.y) + ", " +
                              std::to_string(size.z) + " angstroms";
    _modelDescriptor =
        std::make_shared<ModelDescriptor>(std::move(model), _descriptor.name,
                                          header, metadata);
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
                brayns::PropertyMap props;
                props.setProperty(
                    {MATERIAL_PROPERTY_SHADING_MODE,
                     static_cast<int>(MaterialShadingMode::basic)});
                props.setProperty({MATERIAL_PROPERTY_USER_PARAMETER, 1.0});

                RGBColor rgb{255, 255, 255};
                const auto it = atomColorMap.find(atom.second.element);
                if (it != atomColorMap.end())
                    rgb = (*it).second;

                material->setDiffuseColor(
                    {rgb.r / 255.f, rgb.g / 255.f, rgb.b / 255.f});
                material->updateProperties(props);
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

} // namespace bioexplorer
