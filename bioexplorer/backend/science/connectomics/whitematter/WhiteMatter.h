/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
 *
 * Copyright 2020-2024 Blue BrainProject / EPFL
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
/**
 * Load whitematter from database
 */
class WhiteMatter : public common::SDFGeometries
{
public:
    /**
     * @brief Construct a new WhiteMatter object
     *
     * @param scene 3D scene into which the white matter should be loaded
     * @param details Set of attributes defining how the whitematter should be
     * loaded
     */
    WhiteMatter(core::Scene& scene, const details::WhiteMatterDetails& details, const core::Vector3d& position,
                const core::Quaterniond& rotation, const core::LoaderProgress& callback = core::LoaderProgress());

private:
    double _getDisplacementValue(const DisplacementElement& element) final { return 0; }

    void _buildModel(const core::LoaderProgress& callback);

    const details::WhiteMatterDetails _details;
    core::Scene& _scene;
};
} // namespace connectomics
} // namespace bioexplorer
