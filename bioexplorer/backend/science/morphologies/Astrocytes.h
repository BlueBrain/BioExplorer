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
