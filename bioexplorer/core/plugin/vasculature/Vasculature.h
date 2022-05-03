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
const double segmentDisplacementFrequency = 1.0;

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
     * @brief Apply a specified color scheme to the vasculature
     *
     * @param details Details of the color scheme to apply
     */
    void setColorScheme(const VasculatureColorSchemeDetails& details);

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
    uint64_t getNbNodes() const { return _nodes.size(); }

    /**
     * @brief Get the number of sub-graphs in the vasculature
     *
     * @return uint64_t Number of sub-graphs in the vasculature
     */
    uint64_t getNbSubGraphs() const { return _graphs.size(); }

    /**
     * @brief Get the number of pairs in the vasculature
     *
     * @return uint64_t Number of pairs in the vasculature
     */
    uint64_t getNbPairs() const { return _nbPairs; }

    /**
     * @brief Get the number of entry nodes in the vasculature
     *
     * @return uint64_t Number of entry nodes in the vasculature
     */
    uint64_t getNbEntryNodes() const { return _nbEntryNodes; }

    /**
     * @brief Get the number of sections in the vasculature
     *
     * @return uint64_t Number of sections in the vasculature
     */
    uint64_t getNbSections() const { return _sectionIds.size(); }

    /**
     * @brief Get the size of the node population in the vasculature
     *
     * @return uint64_t Size of the node population in the vasculature
     */
    uint64_t getPopulationSize() const { return _populationSize; }

    /**
     * @brief Get the maximum number of segments per section in the vasculature
     *
     * @return uint64_t Maximum number of segments per section in the
     * vasculature
     */
    uint64_t getNbMaxSegmentsPerSection() const
    {
        return _nbMaxSegmentsPerSection;
    }

private:
    void _buildGraphModel(Model& model,
                          const VasculatureColorSchemeDetails& details);
    void _buildSimpleModel(Model& model,
                           const VasculatureColorSchemeDetails& details,
                           const doubles& radii = doubles());
    void _buildAdvancedModel(Model& model,
                             const VasculatureColorSchemeDetails& details,
                             const doubles& radii = doubles());

    void _buildEdges(Model& model);

    void _importFromDB();
    void _buildModel(const VasculatureColorSchemeDetails& details =
                         VasculatureColorSchemeDetails(),
                     const doubles& radii = doubles());

    void _applyPaletteToModel(Model& model, const doubles& palette);

    const VasculatureDetails _details;
    Scene& _scene;
    GeometryNodes _nodes;
    uint64_t _populationSize{0};
    std::set<uint64_t> _graphs;
    std::set<uint64_t> _sectionIds;
    std::map<uint64_t, uint64_ts> _sections;
    uint64_t _nbMaxSegmentsPerSection{0};
    uint64_t _nbPairs{0};
    uint64_t _nbEntryNodes{0};
};
} // namespace vasculature
} // namespace bioexplorer
