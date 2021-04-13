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

#include "Glycans.h"

#include <plugin/common/CommonTypes.h>
#include <plugin/common/Utils.h>

#include <brayns/engineapi/Material.h>
#include <brayns/engineapi/Scene.h>

#include <sstream>

namespace bioexplorer
{
Glycans::Glycans(Scene& scene, const SugarsDetails& sd)
    : Molecule(scene, {})
    , _details(sd)
{
    size_t lineIndex{0};

    std::stringstream lines{_details.contents};
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
    if (_details.recenter)
    {
        // Get current center
        for (const auto& atom : _atomMap)
            bounds.merge(atom.second.position);
        const auto& center = bounds.getCenter();

        const auto firstAtomPosition = _atomMap.begin()->second.position;
        const auto translation = center - firstAtomPosition;
        // Translate according to position of first atom
        for (auto& atom : _atomMap)
            atom.second.position -= translation;
    }

    _buildModel(_details.assemblyName, _details.name, title, header,
                _details.representation, _details.atomRadiusMultiplier,
                _details.loadBonds);
}

} // namespace bioexplorer
