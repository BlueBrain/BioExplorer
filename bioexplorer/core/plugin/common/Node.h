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
using MaterialSet = std::set<uint64_t>;
using Neighbours = std::set<size_t>;

const int64_t NO_USER_DATA = -1;

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
    Node(const double scale = 1.0);

    /**
     * @brief Get the Model Descriptor object
     *
     * @return ModelDescriptorPtr Pointer to the model descriptor
     */
    const ModelDescriptorPtr getModelDescriptor() const;

    void addSDFDemo(Model& model);

protected:
    void _createMaterials(const MaterialSet& materialIds, Model& model);
    void _setMaterialExtraAttributes();
    size_t _addSDFGeometry(SDFMorphologyData& sdfMorphologyData,
                           const SDFGeometry& geometry,
                           const Neighbours& neighbours,
                           const size_t materialId);

    size_t _addSphere(const bool useSDF, const Vector3f& position,
                      const float radius, const size_t materialId,
                      const uint64_t userDataOffset, Model& model,
                      SDFMorphologyData& sdfMorphologyData,
                      const Neighbours& neighbours,
                      const float displacementRatio = 1.f);

    size_t _addCone(const bool useSDF, const Vector3f& position,
                    const float radius, const Vector3f& target,
                    const float previousRadius, const size_t materialId,
                    const uint64_t userDataOffset, Model& model,
                    SDFMorphologyData& sdfMorphologyData,
                    const Neighbours& neighbours,
                    const float displacementRatio = 1.f);

    void _finalizeSDFGeometries(Model& model,
                                SDFMorphologyData& sdfMorphologyData);

    ModelDescriptorPtr _modelDescriptor{nullptr};
    Boxd _bounds;
    uint32_t _uuid;
    double _scale{1.0};
};

typedef std::shared_ptr<Node> NodePtr;
typedef std::map<std::string, NodePtr> NodeMap;

} // namespace common
} // namespace bioexplorer