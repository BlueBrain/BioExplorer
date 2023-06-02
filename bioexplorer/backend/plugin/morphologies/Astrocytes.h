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

#include "Morphologies.h"

#include <plugin/api/Params.h>
#include <plugin/common/Types.h>

namespace bioexplorer
{
namespace morphology
{
using namespace core;
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
    Astrocytes(Scene& scene, const AstrocytesDetails& details, const Vector3d& assemblyPosition,
               const Quaterniond& assemblyRotation);

    /**
     * @brief Apply a vasculature radius report to the astrocyte. This modifies
     * the end-feet of the astrocytes according to the vasculature radii defined
     * in the report
     *
     * @param details Details of the report
     */
    void setVasculatureRadiusReport(const VasculatureRadiusReportDetails& details);

private:
    double _getDisplacementValue(const DisplacementElement& element) final;

    void _logRealismParams();
    void _buildModel(const doubles& radii = doubles());
    void _addEndFoot(ThreadSafeContainer& container, const Vector3d& somaCenter, const EndFootMap& endFeet,
                     const doubles& radii, const size_t materialId);
    void _addMicroDomain(TriangleMesh& mesh, const uint64_t astrocyteId);
    void _buildMicroDomain(ThreadSafeContainer& container, const uint64_t astrocyteId, const size_t materialId);
    const AstrocytesDetails _details;

    double _maxDistanceToSoma{0.0};
    Scene& _scene;
};
} // namespace morphology
} // namespace bioexplorer
