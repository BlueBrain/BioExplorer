/* Copyright (c) 2018-2021, EPFL/Blue Brain Project
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

/**
 * Load vasculature from database
 */
class Vasculature : public SDFGeometries
{
public:
    Vasculature(Scene& scene, const VasculatureDetails& details);

    void setColorScheme(const VasculatureColorSchemeDetails& details);

    void setRadiusReport(const VasculatureRadiusReportDetails& details);

    uint64_t getNbNodes() const { return _nodes.size(); }
    uint64_t getNbSubGraphs() const { return _graphs.size(); }
    uint64_t getNbPairs() const { return _nbPairs; }
    uint64_t getNbEntryNodes() const { return _nbEntryNodes; }
    uint64_t getNbSections() const { return _sectionIds.size(); }
    uint64_t getPopulationSize() const { return _populationSize; }
    uint64_t getNbMaxPointsPerSection() const { return _nbMaxPointsPerSection; }

private:
    void _buildGraphModel(Model& model,
                          const VasculatureColorSchemeDetails& details);
    void _buildSimpleModel(Model& model,
                           const VasculatureColorSchemeDetails& details,
                           const doubles& radii = doubles());
    void _buildAdvancedModel(Model& model,
                             const VasculatureColorSchemeDetails& details,
                             const doubles& radii = doubles());

    void _importFromDB();
    void _buildModel(const VasculatureColorSchemeDetails& details =
                         VasculatureColorSchemeDetails(),
                     const doubles& radii = doubles());
    const VasculatureDetails _details;
    Scene& _scene;
    GeometryNodes _nodes;
    uint64_t _populationSize{0};
    std::set<uint64_t> _graphs;
    std::set<uint64_t> _sectionIds;
    std::map<uint64_t, uint64_ts> _sections;
    uint64_t _nbMaxPointsPerSection{0};
    uint64_t _nbPairs{0};
    uint64_t _nbEntryNodes{0};
};
} // namespace vasculature
} // namespace bioexplorer
