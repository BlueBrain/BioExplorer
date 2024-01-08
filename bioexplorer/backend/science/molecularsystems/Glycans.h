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

#include <platform/core/engineapi/Model.h>
#include <science/api/Params.h>
#include <science/molecularsystems/Molecule.h>

namespace bioexplorer
{
namespace molecularsystems
{
/**
 * @brief The Glycans class
 */
class Glycans : public Molecule
{
public:
    /**
     * @brief Construct a new Glycans object
     *
     * @param scene The 3D scene where the glycans are added
     * @param details The data structure describing the glycans
     */
    Glycans(core::Scene& scene, const details::SugarDetails& details);

private:
    details::SugarDetails _details;
};
} // namespace molecularsystems
} // namespace bioexplorer
