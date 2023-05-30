/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2023 Blue BrainProject / EPFL
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

#pragma once

#include <plugin/common/SDFGeometries.h>

namespace bioexplorer
{
namespace atlas
{
using namespace brayns;
using namespace common;

/**
 * Load brain atlas from database
 */
class Atlas : public SDFGeometries
{
public:
    /**
     * @brief Construct a new Vasculature object
     *
     * @param scene 3D scene into which the vasculature should be loaded
     * @param details Set of attributes defining how the vasculature should be
     * loaded
     */
    Atlas(Scene& scene, const AtlasDetails& details, const Vector3d& position, const Quaterniond& rotation);

private:
    double _getDisplacementValue(const DisplacementElement& element) final { return 0; }

    void _load();

    const AtlasDetails _details;
    Scene& _scene;
};
} // namespace atlas
} // namespace bioexplorer
