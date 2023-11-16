/*
 *
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * This file is part of Blue Brain BioExplorer <https://github.com/BlueBrain/BioExplorer>
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

#include <science/api/Params.h>
#include <science/common/Types.h>

namespace bioexplorer
{
namespace morphology
{
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
    Astrocytes(core::Scene& scene, const details::AstrocytesDetails& details, const core::Vector3d& assemblyPosition,
               const core::Quaterniond& assemblyRotation,
               const core::LoaderProgress& callback = core::LoaderProgress());

    /**
     * @brief Apply a vasculature radius report to the astrocyte. This modifies the end-feet of the astrocytes according
     * to the vasculature radii defined in the report
     *
     * @param details Details of the report
     */
    void setVasculatureRadiusReport(const details::VasculatureRadiusReportDetails& details);

private:
    double _getDisplacementValue(const DisplacementElement& element) final;

    void _logRealismParams();
    void _buildModel(const core::LoaderProgress& callback, const doubles& radii = doubles());
    void _addEndFoot(common::ThreadSafeContainer& container, const core::Vector3d& somaCenter,
                     const EndFootMap& endFeet, const doubles& radii, const size_t materialId);
    void _addMicroDomain(core::TriangleMesh& mesh, const uint64_t astrocyteId);
    void _buildMicroDomain(common::ThreadSafeContainer& container, const uint64_t astrocyteId, const size_t materialId);
    const details::AstrocytesDetails _details;

    double _maxDistanceToSoma{0.0};
    core::Scene& _scene;
};
} // namespace morphology
} // namespace bioexplorer
