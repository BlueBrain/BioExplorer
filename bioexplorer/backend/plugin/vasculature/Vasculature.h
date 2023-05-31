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

#pragma once

#include <plugin/api/Params.h>
#include <plugin/common/SDFGeometries.h>
#include <plugin/common/Types.h>

#include <core/brayns/common/loader/Loader.h>
#include <core/brayns/parameters/GeometryParameters.h>

namespace bioexplorer
{
namespace vasculature
{
using namespace brayns;
using namespace common;

/**
 * Load vasculature from database
 */
class Vasculature : public SDFGeometries
{
public:
    /**
     * @brief Construct a new Vasculature object
     *
     * @param scene 3D scene into which the vasculature should be loaded
     * @param details Set of attributes defining how the vasculature should be
     * loaded
     */
    Vasculature(Scene& scene, const VasculatureDetails& details, const Vector3d& assemblyPosition,
                const Quaterniond& assemblyRotation);

    /**
     * @brief Apply a radius report to the astrocyte. This modifies vasculature
     * structure according to radii defined in the report
     *
     * @param details Details of the report
     */
    void setRadiusReport(const VasculatureRadiusReportDetails& details);

    /**
     * @brief Get the number of nodes in the vasculature
     *
     * @return uint64_t Number of nodes in the vasculature
     */
    uint64_t getNbNodes() const { return _nbNodes; }

private:
    double _getDisplacementValue(const DisplacementElement& element) final;

    void _logRealismParams();

    void _addGraphSection(ThreadSafeContainer& container, const GeometryNode& srcNode, const GeometryNode& dstNode,
                          const size_t materialId);
    void _addSimpleSection(ThreadSafeContainer& container, const GeometryNode& srcNode, const GeometryNode& dstNode,
                           const size_t materialId, const uint64_t userData);
    void _addDetailedSection(ThreadSafeContainer& container, const GeometryNodes& nodes, const size_t baseMaterialId,
                             const doubles& radii, const Vector2d& radiusRange);
    void _addOrientation(ThreadSafeContainer& container, const GeometryNodes& nodes, const uint64_t sectionId);
    void _buildModel(const doubles& radii = doubles());

    const VasculatureDetails _details;
    Scene& _scene;
    uint64_t _nbNodes{0};
};
} // namespace vasculature
} // namespace bioexplorer
