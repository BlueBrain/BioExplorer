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

#include <plugin/api/Params.h>
#include <plugin/common/Node.h>
#include <plugin/common/Types.h>

namespace bioexplorer
{
namespace morphology
{
using namespace brayns;
using namespace common;

using MaterialSet = std::set<uint64_t>;

/**
 * Load Neurons from database
 */
class Neurons : public common::Node
{
public:
    Neurons(Scene& scene, const NeuronsDetails& details);

private:
    void _buildNeuron();

    size_t _getNbMitochondrionSegments() const;

    void _addSection(Model& model, const uint64_t sectionId,
                     const Section& section, const Vector3d& somaPosition,
                     const double somaRadius, const uint64_t sdfSomaGroupId,
                     const size_t baseMaterialId,
                     const double mitochondriaDensity,
                     SDFMorphologyData& sdfMorphologyData, uint64_t& sdfGroupId,
                     MaterialSet& materialIds);

    void _addSomaInternals(const uint64_t index, Model& model,
                           const size_t materialId,
                           const Vector3d& somaPosition,
                           const double somaRadius,
                           const double mitochondriaDensity,
                           SDFMorphologyData& sdfMorphologyData,
                           uint64_t& sdfGroupId);

    void _addSectionInternals(const double sectionLength,
                              const double sectionVolume,
                              const Vector4fs& points,
                              const double mitochondriaDensity,
                              const size_t materialId,
                              SDFMorphologyData& sdfMorphologyData,
                              uint64_t& sdfGroupId, Model& model);

    void _addAxonMyelinSheath(const Vector3d& somaPosition,
                              const double sectionLength,
                              const Vector4fs& points,
                              const double mitochondriaDensity,
                              const size_t materialId,
                              SDFMorphologyData& sdfMorphologyData,
                              uint64_t& sdfGroupId, Model& model);

    const NeuronsDetails _details;
    Scene& _scene;
};
} // namespace morphology
} // namespace bioexplorer
