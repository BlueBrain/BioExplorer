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

#pragma once

#include <brayns/engineapi/Model.h>
#include <plugin/api/Params.h>
#include <plugin/bioexplorer/Molecule.h>
#include <plugin/common/Types.h>

namespace bioexplorer
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
     * @param sd The data structure describing the glycans
     */
    Glycans(Scene& scene, const SugarsDescriptor& sd);

private:
    SugarsDescriptor _descriptor;
};
} // namespace bioexplorer
