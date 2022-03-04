/*
 * The Blue Brain BioExplorer is a tool for scientists to extract and analyse
 * scientific data from visualization
 *
 * Copyright 2020-2022 Blue BrainProject / EPFL
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

#include <plugin/common/Types.h>

namespace bioexplorer
{
namespace common
{
using namespace brayns;

/**
 * @brief The Node class
 */
class Node
{
public:
    /**
     * @brief Construct a new Node object
     *
     */
    Node();

    /**
     * @brief Get the Model Descriptor object
     *
     * @return ModelDescriptorPtr Pointer to the model descriptor
     */
    const ModelDescriptorPtr getModelDescriptor() const;

protected:
    void _setMaterialExtraAttributes();
    size_t _addSDFGeometry(SDFMorphologyData& sdfMorphologyData,
                           const SDFGeometry& geometry,
                           const std::set<size_t>& neighbours,
                           const size_t materialId, const int section);

    void _addStepSphereGeometry(
        const bool useSDF, const Vector3d& position, const double radius,
        const size_t materialId, const uint64_t userDataOffset, Model& model,
        SDFMorphologyData& sdfMorphologyData, const uint32_t sdfGroupId,
        const Vector3f& displacementParams = Vector3f());

    void _addStepConeGeometry(const bool useSDF, const Vector3d& position,
                              const double radius, const Vector3d& target,
                              const double previousRadius,
                              const size_t materialId,
                              const uint64_t userDataOffset, Model& model,
                              SDFMorphologyData& sdfMorphologyData,
                              const uint32_t sdfGroupId,
                              const Vector3f& displacementParams = Vector3f());

    void _finalizeSDFGeometries(Model& model,
                                SDFMorphologyData& sdfMorphologyData);

    ModelDescriptorPtr _modelDescriptor{nullptr};
    Boxd _bounds;
    uint32_t _uuid;
};

typedef std::shared_ptr<Node> NodePtr;
typedef std::map<std::string, NodePtr> NodeMap;

} // namespace common
} // namespace bioexplorer