/*
    Copyright 2020 - 2024 Blue Brain Project / EPFL

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#pragma once

#include <science/common/Node.h>

namespace bioexplorer
{
namespace molecularsystems
{
using ModelInstanceId = std::pair<uint64_t, uint64_t>;

/**
 * @brief An Enzyme reaction is a object that combines an existing enzyme, a list of substrates and a list of products.
 * It implements the way those molecules interact with each other to describe the chemical reaction.
 *
 */
class EnzymeReaction : public common::Node
{
public:
    /**
     * @brief Construct a new EnzymeReaction object
     *
     * @param scene The 3D scene where the membrane are added
     * @param details Details about the enzyme reaction
     * @param enzymeAssembly Pointer to the assembly containing the enzyme
     * @param enzyme Pointer to the enzyme
     * @param substrates List of pointers to the substrates
     * @param products List of pointers to the products
     */
    EnzymeReaction(core::Scene& scene, const details::EnzymeReactionDetails& details,
                   common::AssemblyPtr enzymeAssembly, ProteinPtr enzyme, Proteins& substrates, Proteins& products);

    /**
     * @brief Set the progress of the reaction process
     *
     * @param instanceId Instance identifier of the enzyme protein
     * @param progress Progress of the reaction betweem 0 and 1
     */
    void setProgress(const uint64_t instanceId, const double progress);

protected:
    core::Quaterniond _getMoleculeRotation(const double progress, const double rotationSpeed = 5.0) const;

    core::Scene& _scene;
    common::AssemblyPtr _enzymeAssembly;
    ProteinPtr _enzyme{nullptr};
    Proteins _substrates;
    Proteins _products;
    std::map<ModelInstanceId, core::Transformation> _enzymeInitialTransformations;
    std::map<ModelInstanceId, core::Transformation> _substrateInitialTransformations;
    std::map<ModelInstanceId, core::Transformation> _productInitialTransformations;
    const details::EnzymeReactionDetails& _details;
};
} // namespace molecularsystems
} // namespace bioexplorer
