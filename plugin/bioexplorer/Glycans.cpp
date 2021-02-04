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

#include "Glycans.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Scene.h>

#include <sstream>

namespace bioexplorer
{
Glycans::Glycans(Scene& scene, const SugarsDescriptor& sd)
    : Molecule(scene, {})
    , _descriptor(sd)
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

    _buildModel(_descriptor.assemblyName, _descriptor.name, title, header,
                _descriptor.representation, _descriptor.atomRadiusMultiplier,
                _descriptor.loadBonds);
}

} // namespace bioexplorer
