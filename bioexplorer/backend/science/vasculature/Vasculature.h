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

#pragma once

#include <science/api/Params.h>
#include <science/common/SDFGeometries.h>
#include <science/common/Types.h>

#include <platform/core/common/loader/Loader.h>
#include <platform/core/parameters/GeometryParameters.h>

namespace bioexplorer
{
namespace vasculature
{
/**
 * Load vasculature from database
 */
class Vasculature : public common::SDFGeometries
{
public:
    /**
     * @brief Construct a new Vasculature object
     *
     * @param scene 3D scene into which the vasculature should be loaded
     * @param details Set of attributes defining how the vasculature should be
     * loaded
     */
    Vasculature(core::Scene& scene, const details::VasculatureDetails& details, const core::Vector3d& assemblyPosition,
                const core::Quaterniond& assemblyRotation,
                const core::LoaderProgress& callback = core::LoaderProgress());

    /**
     * @brief Apply a radius report to the astrocyte. This modifies vasculature
     * structure according to radii defined in the report
     *
     * @param details Details of the report
     */
    void setRadiusReport(const details::VasculatureRadiusReportDetails& details);

    /**
     * @brief Get the number of nodes in the vasculature
     *
     * @return uint64_t Number of nodes in the vasculature
     */
    uint64_t getNbNodes() const { return _nbNodes; }

private:
    double _getDisplacementValue(const DisplacementElement& element) final;

    void _logRealismParams();

    void _addGraphSection(common::ThreadSafeContainer& container, const common::GeometryNode& srcNode,
                          const common::GeometryNode& dstNode, const size_t materialId);
    void _addSimpleSection(common::ThreadSafeContainer& container, const common::GeometryNode& srcNode,
                           const common::GeometryNode& dstNode, const size_t materialId, const uint64_t userData);
    void _addDetailedSection(common::ThreadSafeContainer& container, const common::GeometryNodes& nodes,
                             const size_t baseMaterialId, const doubles& radii, const core::Vector2d& radiusRange);
    void _addOrientation(common::ThreadSafeContainer& container, const common::GeometryNodes& nodes,
                         const uint64_t sectionId);
    void _buildModel(const core::LoaderProgress& callback, const doubles& radii = doubles());

    const details::VasculatureDetails _details;
    core::Scene& _scene;
    uint64_t _nbNodes{0};
};
} // namespace vasculature
} // namespace bioexplorer
