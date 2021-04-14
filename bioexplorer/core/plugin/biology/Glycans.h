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

#pragma once

#include <brayns/engineapi/Model.h>
#include <plugin/api/Params.h>
#include <plugin/biology/Molecule.h>
#include <plugin/common/Types.h>

namespace bioexplorer
{
namespace biology
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
    Glycans(Scene& scene, const SugarsDetails& details);

private:
    SugarsDetails _details;
};
} // namespace biology
} // namespace bioexplorer
