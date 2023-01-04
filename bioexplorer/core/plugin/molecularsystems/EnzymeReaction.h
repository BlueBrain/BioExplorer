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

#include <plugin/common/Node.h>

namespace bioexplorer
{
namespace molecularsystems
{
using namespace brayns;
using namespace common;
using namespace details;

using ModelInstanceId = std::pair<uint64_t, uint64_t>;

/**
 * @brief An Enzyme reaction is a object that combines an existing enyzme, a
 * list of substrates and a list of products. It implements the way those
 * molecules interact with each other to describe the chemical reaction.
 *
 */
class EnzymeReaction : public Node
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
    EnzymeReaction(Scene& scene, const EnzymeReactionDetails& details,
                   AssemblyPtr enzymeAssembly, ProteinPtr enzyme,
                   Proteins& substrates, Proteins& products);

    /**
     * @brief Set the progress of the reaction process
     *
     * @param instanceId Instance identifier of the enzyme protein
     * @param progress Progress of the reaction betweem 0 and 1
     */
    void setProgress(const uint64_t instanceId, const double progress);

protected:
    Quaterniond _getMoleculeRotation(const double progress,
                                     const double rotationSpeed = 5.0) const;

    Scene& _scene;
    AssemblyPtr _enzymeAssembly;
    ProteinPtr _enzyme{nullptr};
    Proteins _substrates;
    Proteins _products;
    std::map<ModelInstanceId, Transformation> _enzymeInitialTransformations;
    std::map<ModelInstanceId, Transformation> _substrateInitialTransformations;
    std::map<ModelInstanceId, Transformation> _productInitialTransformations;
    const EnzymeReactionDetails& _details;
};
} // namespace molecularsystems
} // namespace bioexplorer
