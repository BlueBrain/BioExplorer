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

#include "Types.h"

#include <platform/core/common/utils/EnumUtils.h>

namespace core
{
template <>
inline std::vector<std::pair<std::string, bioexplorer::details::VasculatureColorScheme>> enumMap()
{
    return {{"None", bioexplorer::details::VasculatureColorScheme::none},
            {"Node", bioexplorer::details::VasculatureColorScheme::node},
            {"Section", bioexplorer::details::VasculatureColorScheme::section},
            {"Sub-graph", bioexplorer::details::VasculatureColorScheme::subgraph},
            {"Pair", bioexplorer::details::VasculatureColorScheme::pair},
            {"Entry node", bioexplorer::details::VasculatureColorScheme::entry_node},
            {"Radius", bioexplorer::details::VasculatureColorScheme::radius},
            {"Section point", bioexplorer::details::VasculatureColorScheme::section_points},
            {"Section orientation", bioexplorer::details::VasculatureColorScheme::section_orientation},
            {"Region", bioexplorer::details::VasculatureColorScheme::region}};
}

template <>
inline std::vector<std::pair<std::string, bioexplorer::details::VasculatureRealismLevel>> enumMap()
{
    return {{"None", bioexplorer::details::VasculatureRealismLevel::none},
            {"Section", bioexplorer::details::VasculatureRealismLevel::section},
            {"Bifurcation", bioexplorer::details::VasculatureRealismLevel::bifurcation},
            {"All", bioexplorer::details::VasculatureRealismLevel::all}};
}

template <>
inline std::vector<std::pair<std::string, bioexplorer::details::VasculatureRepresentation>> enumMap()
{
    return {{"Graph", bioexplorer::details::VasculatureRepresentation::graph},
            {"Section", bioexplorer::details::VasculatureRepresentation::section},
            {"Segment", bioexplorer::details::VasculatureRepresentation::segment},
            {"Optimized segment", bioexplorer::details::VasculatureRepresentation::optimized_segment},
            {"Bezier", bioexplorer::details::VasculatureRepresentation::bezier}};
}

template <>
inline std::vector<std::pair<std::string, bioexplorer::morphology::PopulationColorScheme>> enumMap()
{
    return {{"None", bioexplorer::morphology::PopulationColorScheme::none},
            {"Id", bioexplorer::morphology::PopulationColorScheme::id}};
}

template <>
inline std::vector<std::pair<std::string, bioexplorer::morphology::MorphologyColorScheme>> enumMap()
{
    return {{"None", bioexplorer::morphology::MorphologyColorScheme::none},
            {"Section type", bioexplorer::morphology::MorphologyColorScheme::section_type},
            {"Section orientation", bioexplorer::morphology::MorphologyColorScheme::section_orientation},
            {"Distance to soma", bioexplorer::morphology::MorphologyColorScheme::distance_to_soma}};
}

template <>
inline std::vector<std::pair<std::string, bioexplorer::morphology::MorphologyRepresentation>> enumMap()
{
    return {{"Graph", bioexplorer::morphology::MorphologyRepresentation::graph},
            {"Section", bioexplorer::morphology::MorphologyRepresentation::section},
            {"Segment", bioexplorer::morphology::MorphologyRepresentation::segment},
            {"Orientation", bioexplorer::morphology::MorphologyRepresentation::orientation},
            {"Bezier", bioexplorer::morphology::MorphologyRepresentation::bezier},
            {"Contour", bioexplorer::morphology::MorphologyRepresentation::contour},
            {"Surface", bioexplorer::morphology::MorphologyRepresentation::surface}};
}

template <>
inline std::vector<std::pair<std::string, bioexplorer::morphology::MorphologyRealismLevel>> enumMap()
{
    return {{"None", bioexplorer::morphology::MorphologyRealismLevel::none},
            {"Soma", bioexplorer::morphology::MorphologyRealismLevel::soma},
            {"Axon", bioexplorer::morphology::MorphologyRealismLevel::axon},
            {"Dendrite", bioexplorer::morphology::MorphologyRealismLevel::dendrite},
            {"Internals", bioexplorer::morphology::MorphologyRealismLevel::internals},
            {"Externals", bioexplorer::morphology::MorphologyRealismLevel::externals},
            {"Spine", bioexplorer::morphology::MorphologyRealismLevel::spine},
            {"End foot", bioexplorer::morphology::MorphologyRealismLevel::end_foot},
            {"All", bioexplorer::morphology::MorphologyRealismLevel::all}};
}

template <>
inline std::vector<std::pair<std::string, bioexplorer::morphology::MorphologySynapseType>> enumMap()
{
    return {{"None", bioexplorer::morphology::MorphologySynapseType::none},
            {"Afferent", bioexplorer::morphology::MorphologySynapseType::afferent},
            {"Efferent", bioexplorer::morphology::MorphologySynapseType::efferent},
            {"Debug", bioexplorer::morphology::MorphologySynapseType::debug},
            {"All", bioexplorer::morphology::MorphologySynapseType::all}};
}
} // namespace core
