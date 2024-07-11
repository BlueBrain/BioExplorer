/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "Glycans.h"

#include <science/common/Utils.h>

#include <platform/core/engineapi/Material.h>
#include <platform/core/engineapi/Scene.h>

using namespace core;

namespace bioexplorer
{
using namespace common;
using namespace details;

namespace molecularsystems
{
Glycans::Glycans(Scene& scene, const SugarDetails& details)
    : Molecule(scene, Vector3d(), doublesToQuaterniond(details.rotation), {})
    , _details(details)
{
    size_t lineIndex{0};

    std::stringstream lines{_details.contents};
    std::string line;
    std::string title;
    std::string header{details.name};

    while (getline(lines, line, '\n'))
    {
        if (line.find(KEY_ATOM) == 0 || line.find(KEY_HETATM) == 0)
            _readAtom(line, true);
        else if (details.loadBonds && line.find(KEY_CONECT) == 0)
            _readConnect(line);
        else if (line.find(KEY_HEADER) == 0)
            header = _readHeader(line);
        else if (line.find(KEY_TITLE) == 0)
            title = _readTitle(line);
    }
    auto model = scene.createModel();

    if (_details.recenter)
    {
        // Recenter
        Boxd bounds;

        // Get current center
        for (const auto& atom : _atomMap)
            bounds.merge(atom.second.position);
        const Vector3d center = bounds.getCenter();

        const auto firstAtomPosition = _atomMap.begin()->second.position;
        const auto translation = center - firstAtomPosition;

        // Translate according to position of first atom
        for (auto& atom : _atomMap)
            atom.second.position -= translation;
    }

    // Build 3d models according to atoms positions
    _buildModel(_details.assemblyName, _details.name, _details.pdbId, header, _details.representation,
                _details.atomRadiusMultiplier, _details.loadBonds);
}
} // namespace molecularsystems
} // namespace bioexplorer
