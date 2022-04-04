/* Copyright (c) 2020-2022, EPFL/Blue Brain Project
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

#pragma once

#include "Morphologies.h"

#include <plugin/api/Params.h>
#include <plugin/common/Types.h>

namespace bioexplorer
{
namespace morphology
{
using namespace brayns;
using namespace common;

/**
 * Load a population of astrocytes from the database according to specified
 * parameters
 */
class Astrocytes : public Morphologies
{
public:
    /**
     * @brief Construct a new Astrocytes object
     *
     * @param scene 3D scene into which astrocytes should be loaded
     * @param details Set of attributes defining how astrocytes should be loaded
     */
    Astrocytes(Scene& scene, const AstrocytesDetails& details);

    /**
     * @brief Apply a vasculature radius report to the astrocyte. This modifies
     * the end-feet of the astrocytes according to the vasculature radii defined
     * in the report
     *
     * @param details Details of the report
     */
    void setVasculatureRadiusReport(
        const VasculatureRadiusReportDetails& details);

private:
    void _buildModel(const doubles& radii = doubles());
    void _addEndFoot(ThreadSafeContainer& container, const EndFootMap& endFeet,
                     const doubles& radii, const size_t materialId);
    const AstrocytesDetails _details;
    Scene& _scene;
};
} // namespace morphology
} // namespace bioexplorer
