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

#pragma once

#include <plugin/api/Params.h>
#include <plugin/common/SDFGeometries.h>
#include <plugin/common/Types.h>

#include <brayns/common/loader/Loader.h>
#include <brayns/parameters/GeometryParameters.h>

namespace bioexplorer
{
namespace vasculature
{
using namespace brayns;
using namespace common;

const double segmentDisplacementStrength = 0.25;
const double segmentDisplacementFrequency = 0.4;

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
    Vasculature(Scene& scene, const VasculatureDetails& details);

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

    /**
     * @brief Get the number of sections in the vasculature
     *
     * @return uint64_t Number of sections in the vasculature
     */
    uint64_t getNbSections() const { return _nbSections; }

private:
    void _addGraphSection(ThreadSafeContainer& container,
                          const GeometryNode& srcNode,
                          const GeometryNode& dstNode, const size_t materialId);
    void _addSimpleSection(ThreadSafeContainer& container,
                           const GeometryNode& srcNode,
                           const GeometryNode& dstNode,
                           const size_t materialId);
    void _addDetailedSection(ThreadSafeContainer& container,
                             const GeometryNodes& nodes,
                             const size_t baseMaterialId, const doubles& radii,
                             const Vector2d& radiusRange);
    void _addOrientation(ThreadSafeContainer& container,
                         const GeometryNodes& nodes, const uint64_t sectionId);
    void _buildModel(const doubles& radii = doubles());

    const VasculatureDetails _details;
    Scene& _scene;
    uint64_t _nbNodes{0};
    uint64_t _nbSections{0};
};
} // namespace vasculature
} // namespace bioexplorer
