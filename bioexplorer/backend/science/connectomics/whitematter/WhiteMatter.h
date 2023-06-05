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

#include <science/common/SDFGeometries.h>

namespace bioexplorer
{
namespace connectomics
{
using namespace core;
using namespace common;
using namespace details;

/**
 * Load whitematter from database
 */
class WhiteMatter : public SDFGeometries
{
public:
    /**
     * @brief Construct a new WhiteMatter object
     *
     * @param scene 3D scene into which the white matter should be loaded
     * @param details Set of attributes defining how the whitematter should be
     * loaded
     */
    WhiteMatter(Scene& scene, const WhiteMatterDetails& details, const Vector3d& position, const Quaterniond& rotation);

private:
    double _getDisplacementValue(const DisplacementElement& element) final { return 0; }

    void _buildModel();

    void _addStreamline(ThreadSafeContainer& container, const Vector3fs& points, const uint64_t materialId);

    const WhiteMatterDetails _details;
    Scene& _scene;
};
} // namespace connectomics
} // namespace bioexplorer
