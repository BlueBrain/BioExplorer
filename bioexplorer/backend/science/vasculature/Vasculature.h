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
