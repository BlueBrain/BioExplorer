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
#include <plugin/common/Node.h>
#include <plugin/common/Types.h>

#include <brayns/common/loader/Loader.h>
#include <brayns/parameters/GeometryParameters.h>

namespace bioexplorer
{
namespace vasculature
{
using namespace brayns;
using namespace common;
using namespace geometry;

enum class EdgeType
{
    artery = 0,
    capilarity = 1,
    vein = 2
};

typedef struct
{
    Vector3d startPosition;
    double startRadius;
    Vector3d endPosition;
    double endRadius;
    uint64_t sectionId;
    uint64_t graphId;
    uint64_t type;
    uint64_t pairId{0};
    uint64_t entryNode;
} VasculatureNode;
typedef std::map<uint64_t, VasculatureNode> VasculatureNodes;

/**
 * Load vasculature from H5 file
 */
class Vasculature : public common::Node
{
public:
    Vasculature(Scene& scene, const VasculatureDetails& details);

    void setColorScheme(const VasculatureColorSchemeDetails& details);

    void setRadiusReport(const VasculatureRadiusReportDetails& details);

    uint64_t getNbNodes() const { return _nodes.size(); }
    uint64_t getNbSubGraphs() const { return _graphs.size(); }
    uint64_t getNbPairs() const { return _nbPairs; }
    uint64_t getNbSections() const { return _sectionIds.size(); }
    uint64_t getPopulationSize() const { return _populationSize; }

private:
    void _importFromFile();
    void _buildModel(const VasculatureColorSchemeDetails& details =
                         VasculatureColorSchemeDetails());
    size_t _addSDFGeometry(SDFMorphologyData& sdfMorphologyData,
                           const SDFGeometry& geometry,
                           const std::set<size_t>& neighbours,
                           const size_t materialId, const int section);

    void _addStepConeGeometry(
        const bool useSDF, const Vector3d& position, const double radius,
        const Vector3d& target, const double previousRadius,
        const size_t materialId, const uint64_t& userDataOffset, Model& model,
        SDFMorphologyData& sdfMorphologyData, const uint32_t sdfGroupId,
        const Vector3f& displacementParams = Vector3f(0.f));

    void _finalizeSDFGeometries(Model& model,
                                SDFMorphologyData& sdfMorphologyData);

    const VasculatureDetails _details;
    Scene& _scene;
    VasculatureNodes _nodes;
    uint64_t _populationSize{0};
    std::set<uint64_t> _graphs;
    std::set<uint64_t> _sectionIds;
    std::map<uint64_t, uint64_ts> _sections;
    uint64_t _nbPairs{0};
};
} // namespace vasculature
} // namespace bioexplorer
